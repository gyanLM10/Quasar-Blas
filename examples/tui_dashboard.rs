//! Quasar-BLAS Interactive TUI Dashboard
//!
//! A multi-pane terminal dashboard for visualizing GEMM performance across
//! all engine tiers. Uses ratatui for rendering and crossterm for event handling.
//!
//! ## Architecture (Decoupled Render/Sample)
//!
//! - **Render loop**: 60Hz via crossterm event polling (16ms tick)
//! - **Benchmark thread**: Runs GEMM at 500ms intervals, sends results via mpsc
//! - **Data flow**: `BenchmarkResult` → `mpsc::channel` → TUI `try_recv()` per frame
//!
//! Run with: `cargo run --example tui_dashboard --features "tui,gpu"`

use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Bar, BarChart, BarGroup, Block, Borders, Gauge, Paragraph, Sparkline, Wrap,
    },
    Frame, Terminal,
};

use quasar_blas::GemmEngine;
use quasar_blas::cpu::{NaiveGemm, SimdGemm, TiledGemm};
use quasar_blas::gpu::{GpuGemm, ShaderVariant};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineType {
    CpuNaive,
    CpuTiled,
    CpuSimd,
    GpuNaive,
    GpuTiled,
}

impl EngineType {
    const ALL: [EngineType; 5] = [
        EngineType::CpuNaive,
        EngineType::CpuTiled,
        EngineType::CpuSimd,
        EngineType::GpuNaive,
        EngineType::GpuTiled,
    ];

    fn label(&self) -> &'static str {
        match self {
            EngineType::CpuNaive => "CPU Naive",
            EngineType::CpuTiled => "CPU Tiled",
            EngineType::CpuSimd  => "CPU SIMD",
            EngineType::GpuNaive => "GPU Naive",
            EngineType::GpuTiled => "GPU Tiled",
        }
    }

    fn color(&self) -> Color {
        match self {
            EngineType::CpuNaive => Color::Rgb(255, 99, 71),   // Tomato red
            EngineType::CpuTiled => Color::Rgb(255, 165, 0),    // Orange
            EngineType::CpuSimd  => Color::Rgb(0, 206, 209),    // Dark turquoise
            EngineType::GpuNaive => Color::Rgb(138, 43, 226),   // Blue violet
            EngineType::GpuTiled => Color::Rgb(50, 205, 50),    // Lime green
        }
    }
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    engine: EngineType,
    size: usize,
    gflops: f64,
    duration_us: u64,
}

struct App {
    size: usize,
    sizes: Vec<usize>,
    size_index: usize,
    results: Vec<BenchmarkResult>,
    gflops_history: Vec<u64>,
    selected_engine: usize,
    running: bool,
    status: String,
    gpu_info: String,
    should_quit: bool,
    run_all: bool,
}

impl App {
    fn new(gpu_info: String) -> Self {
        Self {
            size: 128,
            sizes: vec![32, 64, 128, 256, 512, 1024],
            size_index: 2,
            results: Vec::new(),
            gflops_history: Vec::new(),
            selected_engine: 2, // Default to CPU SIMD
            running: false,
            status: String::from("Ready. Press [Space] to benchmark, [a] for all engines."),
            gpu_info,
            should_quit: false,
            run_all: false,
        }
    }

    fn increase_size(&mut self) {
        if self.size_index < self.sizes.len() - 1 {
            self.size_index += 1;
            self.size = self.sizes[self.size_index];
        }
    }

    fn decrease_size(&mut self) {
        if self.size_index > 0 {
            self.size_index -= 1;
            self.size = self.sizes[self.size_index];
        }
    }

    fn next_engine(&mut self) {
        self.selected_engine = (self.selected_engine + 1) % EngineType::ALL.len();
    }

    fn peak_gflops(&self) -> f64 {
        self.results
            .iter()
            .map(|r| r.gflops)
            .fold(0.0f64, f64::max)
            .max(1.0) // Avoid division by zero
    }
}

// ───────────────────────── Benchmark Worker ──────────────────────

fn run_cpu_benchmark(engine_type: EngineType, size: usize) -> BenchmarkResult {
    let m = size;
    let k = size;
    let n = size;

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 200) as f32 * 0.01)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 13 + 5) % 200) as f32 * 0.01)
        .collect();
    let mut c = vec![0.0f32; m * n];

    let start = Instant::now();

    match engine_type {
        EngineType::CpuNaive => NaiveGemm.gemm(m, k, n, &a, k, &b, n, &mut c, n).unwrap(),
        EngineType::CpuTiled => TiledGemm::<64>.gemm(m, k, n, &a, k, &b, n, &mut c, n).unwrap(),
        EngineType::CpuSimd => SimdGemm::<64>.gemm(m, k, n, &a, k, &b, n, &mut c, n).unwrap(),
        _ => panic!("Not a CPU engine"),
    }

    let elapsed = start.elapsed();
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    BenchmarkResult {
        engine: engine_type,
        size,
        gflops,
        duration_us: elapsed.as_micros() as u64,
    }
}

fn run_gpu_benchmark(
    engine_type: EngineType,
    size: usize,
    gpu_gemm_naive: &GpuGemm,
    gpu_gemm_tiled: &GpuGemm,
) -> BenchmarkResult {
    let m = size;
    let k = size;
    let n = size;

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 200) as f32 * 0.01)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 13 + 5) % 200) as f32 * 0.01)
        .collect();
    let mut c = vec![0.0f32; m * n];

    let start = Instant::now();

    match engine_type {
        EngineType::GpuNaive => gpu_gemm_naive.gemm(m, k, n, &a, k, &b, n, &mut c, n).unwrap(),
        EngineType::GpuTiled => gpu_gemm_tiled.gemm(m, k, n, &a, k, &b, n, &mut c, n).unwrap(),
        _ => panic!("Not a GPU engine"),
    }

    let elapsed = start.elapsed();
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    BenchmarkResult {
        engine: engine_type,
        size,
        gflops,
        duration_us: elapsed.as_micros() as u64,
    }
}

// ───────────────────────────── UI Rendering ─────────────────────

fn ui(f: &mut Frame, app: &App) {
    // Main layout: top (main area) + bottom (stats bar)
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),     // Main area
            Constraint::Length(5),   // Stats bar
        ])
        .split(f.area());

    // Main area: left sidebar + right content
    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(30),  // Controls sidebar
            Constraint::Min(40),    // Content area
        ])
        .split(outer[0]);

    render_controls(f, main[0], app);
    render_content(f, main[1], app);
    render_stats(f, outer[1], app);
}

fn render_controls(f: &mut Frame, area: Rect, app: &App) {
    let selected = EngineType::ALL[app.selected_engine];

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  [↑/↓] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(format!("Size: {}", app.size)),
        ]),
        Line::from(vec![
            Span::styled("  [Tab]  ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("Switch Engine"),
        ]),
        Line::from(vec![
            Span::styled("  [Spc]  ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("Run Single"),
        ]),
        Line::from(vec![
            Span::styled("  [a]    ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("Run All"),
        ]),
        Line::from(vec![
            Span::styled("  [q]    ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("Quit"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  Engine: "),
            Span::styled(
                selected.label(),
                Style::default().fg(selected.color()).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" ●", Style::default().fg(selected.color())),
        ]),
        Line::from(format!("  Matrix: {}×{}", app.size, app.size)),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                if app.running { "  ◉ Running..." } else { "  ○ Ready" },
                Style::default().fg(if app.running { Color::Yellow } else { Color::Green }),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!("  GPU: {}", app.gpu_info),
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let block = Block::default()
        .title("  ⚡ Controls ")
        .title_style(Style::default().fg(Color::Rgb(0, 206, 209)).add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(60, 60, 80)));

    let paragraph = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
    f.render_widget(paragraph, area);
}

fn render_content(f: &mut Frame, area: Rect, app: &App) {
    // Split content into bar chart + sparkline
    let content = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),    // Bar chart
            Constraint::Length(6),  // Sparkline
        ])
        .split(area);

    render_bar_chart(f, content[0], app);
    render_sparkline(f, content[1], app);
}

fn render_bar_chart(f: &mut Frame, area: Rect, app: &App) {
    let bars: Vec<Bar> = EngineType::ALL
        .iter()
        .map(|engine_type| {
            let gflops = app
                .results
                .iter()
                .rev()
                .find(|r| r.engine == *engine_type)
                .map(|r| r.gflops)
                .unwrap_or(0.0);

            Bar::default()
                .value((gflops * 100.0) as u64) // Scale for visibility
                .label(Line::from(engine_type.label()))
                .style(Style::default().fg(engine_type.color()))
                .text_value(format!("{:.1}", gflops))
        })
        .collect();

    let chart = BarChart::default()
        .block(
            Block::default()
                .title("  📊 GFLOPS by Engine ")
                .title_style(Style::default().fg(Color::Rgb(50, 205, 50)).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(60, 60, 80))),
        )
        .data(BarGroup::default().bars(&bars))
        .bar_width(12)
        .bar_gap(2)
        .direction(Direction::Vertical);

    f.render_widget(chart, area);
}

fn render_sparkline(f: &mut Frame, area: Rect, app: &App) {
    let data: Vec<u64> = if app.gflops_history.is_empty() {
        vec![0; 50]
    } else {
        app.gflops_history.clone()
    };

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title("  📈 Throughput History (GFLOPS) ")
                .title_style(Style::default().fg(Color::Rgb(255, 165, 0)).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(60, 60, 80))),
        )
        .data(&data)
        .style(Style::default().fg(Color::Rgb(0, 206, 209)));

    f.render_widget(sparkline, area);
}

fn render_stats(f: &mut Frame, area: Rect, app: &App) {
    let peak = app.peak_gflops();

    let latest = app.results.last();
    let (time_str, gflops_str, eff_str) = match latest {
        Some(r) => {
            let time = if r.duration_us > 1000 {
                format!("{:.2}ms", r.duration_us as f64 / 1000.0)
            } else {
                format!("{}μs", r.duration_us)
            };
            let efficiency = (r.gflops / peak * 100.0).min(100.0);
            (
                time,
                format!("{:.2}", r.gflops),
                format!("{:.0}%", efficiency),
            )
        }
        None => ("--".into(), "--".into(), "--%".into()),
    };

    let gauge_ratio = latest
        .map(|r| (r.gflops / peak).min(1.0))
        .unwrap_or(0.0);

    // Split stats bar into text + gauge
    let stats_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60),
            Constraint::Percentage(40),
        ])
        .split(area);

    let stats_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Time: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&time_str, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw("  │  "),
            Span::styled("GFLOPS: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&gflops_str, Style::default().fg(Color::Rgb(0, 206, 209)).add_modifier(Modifier::BOLD)),
            Span::raw("  │  "),
            Span::styled("Efficiency: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&eff_str, Style::default().fg(Color::Rgb(50, 205, 50)).add_modifier(Modifier::BOLD)),
            Span::raw("  │  "),
            Span::styled("Peak: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.2}", peak), Style::default().fg(Color::Rgb(255, 165, 0)).add_modifier(Modifier::BOLD)),
        ]),
    ];

    let stats_paragraph = Paragraph::new(stats_text)
        .block(
            Block::default()
                .title("  🔬 Hardware Stats ")
                .title_style(Style::default().fg(Color::Rgb(255, 99, 71)).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(60, 60, 80))),
        );

    let gauge = Gauge::default()
        .block(
            Block::default()
                .title("  Efficiency ")
                .title_style(Style::default().fg(Color::Rgb(138, 43, 226)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(60, 60, 80))),
        )
        .gauge_style(Style::default().fg(Color::Rgb(50, 205, 50)).bg(Color::Rgb(30, 30, 40)))
        .ratio(gauge_ratio)
        .label(format!("{:.0}%", gauge_ratio * 100.0));

    f.render_widget(stats_paragraph, stats_layout[0]);
    f.render_widget(gauge, stats_layout[1]);
}

// ───────────────────────────── Main ─────────────────────────────

fn main() -> io::Result<()> {
    // Initialize GPU engines (one-time cost)
    eprintln!("Initializing GPU...");
    let gpu_naive = GpuGemm::new(ShaderVariant::Naive);
    let gpu_info = gpu_naive.adapter_info();
    let gpu_tiled = GpuGemm::new(ShaderVariant::Tiled);

    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::new(gpu_info);

    // Benchmark result channel (500ms sampling from background thread)
    let (tx, rx) = mpsc::channel::<BenchmarkResult>();

    // Main event loop (60Hz render)
    let tick_rate = Duration::from_millis(16); // ~60 FPS

    loop {
        // Render
        terminal.draw(|f| ui(f, &app))?;

        // Poll for events
        if event::poll(tick_rate)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            app.should_quit = true;
                        }
                        KeyCode::Up => app.increase_size(),
                        KeyCode::Down => app.decrease_size(),
                        KeyCode::Tab => app.next_engine(),
                        KeyCode::Char(' ') if !app.running => {
                            // Run single engine benchmark on background thread
                            app.running = true;
                            app.status = format!("Running {}...", EngineType::ALL[app.selected_engine].label());
                            let engine = EngineType::ALL[app.selected_engine];
                            let size = app.size;
                            let tx = tx.clone();

                            // Run benchmark
                            // GPU operations need to happen on the main thread on macOS
                            if engine == EngineType::GpuNaive || engine == EngineType::GpuTiled {
                                let result = run_gpu_benchmark(engine, size, &gpu_naive, &gpu_tiled);
                                let _ = tx.send(result);
                            } else {
                                std::thread::spawn(move || {
                                    let result = run_cpu_benchmark(engine, size);
                                    let _ = tx.send(result);
                                });
                            }
                        }
                        KeyCode::Char('a') if !app.running => {
                            // Run all engines
                            app.running = true;
                            app.run_all = true;
                            app.status = "Running all engines...".into();
                            let size = app.size;
                            let tx = tx.clone();

                            // Run GPU on main thread
                            let _ = tx.send(run_gpu_benchmark(EngineType::GpuNaive, size, &gpu_naive, &gpu_tiled));
                            let _ = tx.send(run_gpu_benchmark(EngineType::GpuTiled, size, &gpu_naive, &gpu_tiled));

                            // Run CPU in background
                            std::thread::spawn(move || {
                                let _ = tx.send(run_cpu_benchmark(EngineType::CpuNaive, size));
                                let _ = tx.send(run_cpu_benchmark(EngineType::CpuTiled, size));
                                let _ = tx.send(run_cpu_benchmark(EngineType::CpuSimd, size));
                            });
                        }
                        _ => {}
                    }
                }
            }
        }

        // Check for benchmark results (non-blocking)
        while let Ok(result) = rx.try_recv() {
            app.gflops_history.push((result.gflops * 100.0) as u64);
            if app.gflops_history.len() > 50 {
                app.gflops_history.remove(0);
            }
            app.status = format!(
                "{}: {:.2} GFLOPS ({:.2}ms) @ {}×{}",
                result.engine.label(),
                result.gflops,
                result.duration_us as f64 / 1000.0,
                result.size,
                result.size,
            );
            app.results.push(result);
            app.running = false;
        }

        if app.should_quit {
            break;
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Print final summary
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║         Quasar-BLAS Benchmark Summary        ║");
    println!("╠══════════════════════════════════════════════╣");
    for result in &app.results {
        println!(
            "║  {:12} │ {:4}×{:<4} │ {:8.2} GFLOPS  ║",
            result.engine.label(), result.size, result.size, result.gflops
        );
    }
    println!("╚══════════════════════════════════════════════╝");

    Ok(())
}
