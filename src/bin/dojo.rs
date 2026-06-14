use candle_core::Device;
use candle_core::safetensors::load;
use clap::Parser;
use std::path::PathBuf;
use z048::Rater;
use z048::Train;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    rater: PathBuf,
    #[command(flatten)]
    train: Train,
}

fn main() {
    let args = Args::parse();
    eprintln!("loading {}", args.rater.display());
    let rater = Rater::from(load(&args.rater, &Device::Cpu).expect("load checkpoint"));
    args.train.train_iter(&rater, |round, mut finals, losses| {
        let k = (args.train.train_steps / 10).clamp(1, losses.len());
        let l0 = losses[..k].iter().sum::<f32>() / k as f32;
        let l1 = losses[losses.len() - k..].iter().sum::<f32>() / k as f32;
        rater.save(&args.rater);
        finals.sort_by(f64::total_cmp);
        let mean = finals.iter().sum::<f64>() / finals.len() as f64;
        let median = finals[finals.len() / 2];
        eprintln!("round {round}: phi_final mean {mean:.3} median {median:.3} loss {l0:.5} -> {l1:.5}");
    });
}
