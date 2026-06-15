use clap::Parser;
use serde_json::from_reader;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use z048::Rater;
use z048::Train;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    rater: PathBuf,
    #[arg(long)]
    train: PathBuf,
}

fn main() {
    let args = Args::parse();
    eprintln!("loading {}", args.rater.display());
    let file = File::open(&args.train).expect("open train config");
    let trains: Vec<Train> = from_reader(BufReader::new(file)).expect("parse train config");
    let rater = Rater::from(File::open(&args.rater).expect("load checkpoint"));
    for (stage, train) in trains.into_iter().enumerate() {
        train.train(&rater, |round, losses| {
            let k = (losses.len() / 4).max(1);
            let l0 = losses[..k].iter().sum::<f32>() / k as f32;
            let l1 = losses[losses.len() - k..].iter().sum::<f32>() / k as f32;
            rater.save(args.rater.with_extension(format!("s{stage}.r{round}.bin")));
            eprintln!("stage {stage} round {round}: loss {l0:.5} -> {l1:.5}");
        });
    }
}
