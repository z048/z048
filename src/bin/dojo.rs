use candle_core::Device;
use candle_core::Var;
use candle_nn::VarMap;
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
    let varmap = VarMap::new();
    eprintln!("loading {}", args.rater.display());
    {
        let mut data = varmap.data().lock().expect("lock varmap data");
        for (name, t) in candle_core::safetensors::load(&args.rater, &candle_core::Device::Cpu).expect("load checkpoint safetensors") {
            data.insert(name, Var::from_tensor(&t).expect("wrap checkpoint tensor as var"));
        }
    }
    let rater = Rater::from((varmap, Device::Cpu));
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
