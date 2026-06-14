use candle_core::Device;
use clap::Parser;
use std::path::PathBuf;
use z048::Rater;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    rater: PathBuf,
    #[arg(long, num_args = 1.., default_values_t = [128usize, 32])]
    hidden: Vec<usize>,
    #[arg(long, default_value_t = 0x2048_2048_2048_2048)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    let rater = Rater::from((args.hidden.clone(), args.seed, Device::Cpu));
    if let Some(dir) = args.rater.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).expect("create checkpoint directory");
        }
    }
    rater.save(&args.rater);
    eprintln!("minted {:?} -> {}", args.hidden, args.rater.display());
}
