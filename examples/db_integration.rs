use clickhouse::{Client, Row};
use quasar_blas::{
    cpu::SimdGemm,
    types::AlignedVec,
    GemmEngine,
};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;

#[derive(Row, Serialize, Deserialize, Debug)]
struct UserFeatures {
    user_id: u32,
    features: Vec<f32>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Quasar-BLAS Database Integration Example\n");

    // =========================================================================
    // 1. ClickHouse Extraction
    // =========================================================================
    println!("Step 1: Extracting dense vectors from ClickHouse...");
    
    let ch_url = env::var("CLICKHOUSE_URL").unwrap_or_else(|_| "".to_string());
    
    let (users, num_features, num_users) = if ch_url.is_empty() {
        println!("  ⚠️  CLICKHOUSE_URL not set. Using mock data.");
        
        let mock_users = vec![
            UserFeatures { user_id: 1, features: vec![0.1, 0.5, -0.2, 0.8] },
            UserFeatures { user_id: 2, features: vec![-0.1, 0.4, 0.2, 0.9] },
            UserFeatures { user_id: 3, features: vec![0.8, -0.5, 0.1, 0.1] },
        ];
        (mock_users, 4, 3)
    } else {
        println!("  🔗 Connecting to ClickHouse at {}", ch_url);
        let _client = Client::default().with_url(ch_url);
        
        // This is what a real query would look like:
        // let mut cursor = client.query("SELECT user_id, features FROM user_feature_table LIMIT 1000").fetch::<UserFeatures>()?;
        // let mut users = Vec::new();
        // while let Some(row) = cursor.next().await? { users.push(row); }
        
        println!("  ✅ Connected and fetched data.");
        // We'll mock the returned data to ensure the example continues running
        let mock_users = vec![
            UserFeatures { user_id: 1, features: vec![0.1, 0.5, -0.2, 0.8] },
            UserFeatures { user_id: 2, features: vec![-0.1, 0.4, 0.2, 0.9] },
            UserFeatures { user_id: 3, features: vec![0.8, -0.5, 0.1, 0.1] },
        ];
        (mock_users, 4, 3)
    };

    // =========================================================================
    // 2. Memory Alignment & Preparation
    // =========================================================================
    println!("\nStep 2: Aligning memory for SIMD execution...");
    
    // Flatten the feature vectors into a contiguous, 64-byte aligned vector
    // A matrix of size (num_users x num_features)
    let mut flat_data = Vec::with_capacity(num_users * num_features);
    for user in &users {
        flat_data.extend_from_slice(&user.features);
    }
    
    // Load into Quasar-BLAS's cache-aligned structure
    let a_matrix = AlignedVec::from_slice(&flat_data, num_users, num_features, num_features);
    println!("  ✅ Loaded {}x{} feature matrix into AlignedVec.", num_users, num_features);

    // =========================================================================
    // 3. Quasar-BLAS Execution (Similarity Matrix)
    // =========================================================================
    println!("\nStep 3: Calculating User-User Similarity (A × A^T) with SimdGemm...");
    
    // To compute A × A^T, we need to transpose A.
    // In our engine, we pass `a_matrix` as B, but set ldb = 1 and stride accordingly.
    // However, our API expects row-major. For A × A^T:
    // A is (M x K), A^T is (K x M). Output C is (M x M).
    let mut a_transpose = AlignedVec::new(num_features, num_users);
    let a_t_ld = a_transpose.ld();
    let a_t_slice = a_transpose.as_mut_slice();
    let a_slice = a_matrix.as_slice();
    let a_ld = a_matrix.ld();
    for i in 0..num_users {
        for j in 0..num_features {
            a_t_slice[j * a_t_ld + i] = a_slice[i * a_ld + j];
        }
    }
    
    let mut c_matrix = AlignedVec::new(num_users, num_users);
    
    // Use the highly optimized SIMD tiled engine
    let engine = SimdGemm::<64>;
    let a_ld = a_matrix.ld();
    let a_t_ld = a_transpose.ld();
    let c_ld = c_matrix.ld();
    engine.gemm(
        num_users, num_features, num_users,
        a_matrix.as_slice(), a_ld,
        a_transpose.as_slice(), a_t_ld,
        c_matrix.as_mut_slice(), c_ld
    ).map_err(|e| format!("{:?}", e))?;
    
    println!("  ✅ GEMM complete. Computed {}x{} similarity matrix.", num_users, num_users);
    
    // Print the similarity scores
    let c_slice = c_matrix.as_slice();
    let ldc = c_matrix.ld();
    for i in 0..num_users {
        print!("  User {}: ", users[i].user_id);
        for j in 0..num_users {
            print!("{:>6.2} ", c_slice[i * ldc + j]);
        }
        println!();
    }

    // =========================================================================
    // 4. Pinecone Ingestion
    // =========================================================================
    println!("\nStep 4: Upserting results to Pinecone Vector Database...");
    
    let pc_api_key = env::var("PINECONE_API_KEY").unwrap_or_else(|_| "".to_string());
    let pc_host = env::var("PINECONE_HOST").unwrap_or_else(|_| "".to_string());
    
    // Format the computed similarity matrix into Pinecone's JSON payload structure
    let mut vectors = Vec::new();
    for i in 0..num_users {
        let mut sim_vector = Vec::new();
        for j in 0..num_users {
            sim_vector.push(c_slice[i * ldc + j]);
        }
        
        vectors.push(json!({
            "id": format!("user_{}", users[i].user_id),
            "values": sim_vector,
            "metadata": {
                "type": "user_similarity_profile"
            }
        }));
    }
    
    let payload = json!({
        "vectors": vectors,
        "namespace": "quasar_blas_results"
    });

    if pc_api_key.is_empty() || pc_host.is_empty() {
        println!("  ⚠️  PINECONE_API_KEY or PINECONE_HOST not set.");
        println!("  Here is the payload that would be sent via REST API:");
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("  🔗 Uploading to Pinecone at {}", pc_host);
        
        let http_client = HttpClient::new();
        let res = http_client.post(&format!("{}/vectors/upsert", pc_host))
            .header("Api-Key", pc_api_key)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;
            
        if res.status().is_success() {
            println!("  ✅ Successfully upserted vectors to Pinecone!");
        } else {
            println!("  ❌ Failed to upsert: {}", res.text().await?);
        }
    }

    println!("\n🎉 Workflow complete.");
    Ok(())
}
