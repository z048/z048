use candle_core::Device;
use candle_core::safetensors::load;
use clap::Parser;
use rand::Rng;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use z048::Board;
use z048::Dicer;
use z048::Rater;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    slide_rater: PathBuf,
    #[arg(long)]
    spawn_rater: PathBuf,
    #[arg(long, default_value_t = 2)]
    slide_depth: u8,
    #[arg(long, default_value_t = 2)]
    spawn_depth: u8,
    #[arg(long, default_value_t = 128)]
    rounds: u64,
    #[arg(long, default_value_t = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock is before the unix epoch").as_nanos() as u64)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    let slide = Rater::from(load(&args.slide_rater, &Device::Cpu).expect("load checkpoint"));
    let spawn = Rater::from(load(&args.spawn_rater, &Device::Cpu).expect("load checkpoint"));
    let mut results: Vec<(f64, usize, u64)> = Vec::with_capacity(args.rounds as usize);
    for round in 0..args.rounds {
        let mut dicer = Dicer::from(args.seed + round);
        let mut board = Board::from(dicer.r#gen::<u64>());
        let mut plies = 0usize;
        let final_board: Board = loop {
            let sl = slide.sample_slide(board, args.slide_depth, 0.0, &mut dicer);
            let after = board.slide(sl);
            plies += 1;
            let sp = spawn.sample_spawn(after, args.spawn_depth, 0.0, &mut dicer);
            let next = after.spawn(sp);
            if next.end() {
                break next;
            }
            board = next;
        };
        let phi_final = final_board.score();
        let max_rank = <[[u8; 4]; 4]>::from(final_board).into_iter().flatten().map(|r| r as u64).max().expect("board has at least one cell");
        results.push((phi_final, plies, max_rank));
        println!("round {round}: phi_final {phi_final:.3} plies {plies} max_rank {max_rank}");
    }
    let mut phis: Vec<f64> = results.iter().map(|r| r.0).collect();
    phis.sort_by(f64::total_cmp);
    let n = phis.len();
    let mean = phis.iter().sum::<f64>() / n as f64;
    let pct = |p: usize| phis[(n * p / 100).min(n - 1)];
    println!("summary: games {n} phi_final mean {mean:.3} median {:.3} p10 {:.3} p90 {:.3}", pct(50), pct(10), pct(90));
    let top = results.iter().map(|r| r.2).max().expect("at least one game was played");
    let mut hist = String::new();
    for r in 1..=top {
        let c = results.iter().filter(|x| x.2 == r).count();
        if c > 0 {
            hist.push_str(&format!(" {}:{c}", 1u64 << r));
        }
    }
    println!("max-rank histogram (tile:count):{hist}");
}
