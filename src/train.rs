use crate::rater::Rater;
use crate::{Board, Dicer};
use candle_nn::Optimizer;
use candle_nn::ParamsAdamW;
use clap::Parser;
use rand::Rng;

#[derive(Parser, Clone, Copy)]
pub struct Train {
    #[arg(long, default_value_t = 0)]
    pub num_round: usize,
    #[arg(long, default_value_t = 64)]
    pub play_games: usize,
    #[arg(long, default_value_t = 2)]
    pub search_depth: u8,
    #[arg(long, default_value_t = 256)]
    pub train_steps: usize,
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 1_048_576)]
    pub buffer_size: usize,
    #[arg(long, default_value_t = 0x2048_2048_2048_2048)]
    pub random_seed: u64,
    #[arg(long, default_value_t = 0.8)]
    pub td_lambda: f64,
    #[arg(long, default_value_t = 1.0)]
    pub tau_a: f64,
    #[arg(long, default_value_t = 8.0)]
    pub tau_h: f64,
    #[arg(long, default_value_t = 0.02)]
    pub tau_k: f64,
    #[arg(long, default_value_t = 1e-3)]
    pub adamw_lr: f64,
    #[arg(long, default_value_t = 0.9)]
    pub adamw_beta1: f64,
    #[arg(long, default_value_t = 0.999)]
    pub adamw_beta2: f64,
    #[arg(long, default_value_t = 1e-8)]
    pub adamw_eps: f64,
    #[arg(long, default_value_t = 1e-4)]
    pub adamw_wd: f64,
}

impl Train {
    pub fn train_iter(self, rater: &Rater, callback: impl Fn(usize, Vec<f64>, Vec<f32>)) {
        let mut dicer = Dicer::from(self.random_seed);
        let mut opt = rater.adamw(ParamsAdamW { lr: self.adamw_lr, beta1: self.adamw_beta1, beta2: self.adamw_beta2, eps: self.adamw_eps, weight_decay: self.adamw_wd });
        let mut rows: Vec<(Board, f32, f32)> = Vec::new();
        let mut next = 0usize;
        let mut round = 0;
        while self.num_round == 0 || round < self.num_round {
            let mut games = Vec::with_capacity(self.play_games);
            for _ in 0..self.play_games {
                let mut board = Board::from(dicer.random::<u64>());
                let mut afters: Vec<(Board, f64)> = Vec::new();
                let mut befores: Vec<(Board, f64)> = Vec::new();
                let mut ply = 0usize;
                let final_phi = loop {
                    let tau = self.tau_a / (ply as f64 + self.tau_h) + self.tau_k;
                    let before_phi = board.score();

                    let sl = rater.sample_slide(board, self.search_depth, tau, &mut dicer);
                    let after = board.slide(sl);
                    let after_phi = after.score();
                    befores.push((board, after_phi - before_phi));
                    ply += 1;

                    let sp = rater.sample_spawn(after, self.search_depth, tau, &mut dicer);
                    let next_board = after.spawn(sp);
                    let next_phi = next_board.score();
                    afters.push((after, next_phi - after_phi));
                    if next_board.end() {
                        break next_phi;
                    }
                    board = next_board;
                };

                let n = afters.len();
                let lam = self.td_lambda;
                let va = rater.forward(&afters.iter().map(|a| a.0).collect::<Vec<_>>()).to_vec2::<f32>().expect("read v_after column from afterstate eval");
                let vb = rater.forward(&befores.iter().map(|b| b.0).collect::<Vec<_>>()).to_vec2::<f32>().expect("read v_before column from before-state eval");
                let mut put = |row: (Board, f32, f32)| {
                    if rows.len() < self.buffer_size {
                        rows.push(row)
                    } else {
                        rows[next] = row;
                        next = (next + 1) % self.buffer_size
                    }
                };
                let mut g_before = 0.0;
                for t in (0..n).rev() {
                    let g_after = afters[t].1 + if t == n - 1 { 0.0 } else { (1.0 - lam) * vb[t + 1][1] as f64 + lam * g_before };
                    put((afters[t].0, 0.0, g_after as f32));
                    g_before = befores[t].1 + (1.0 - lam) * va[t][0] as f64 + lam * g_after;
                    put((befores[t].0, 1.0, g_before as f32));
                }
                games.push(final_phi);
            }

            let mut losses = Vec::with_capacity(self.train_steps);
            for _ in 0..self.train_steps {
                let idx: Vec<usize> = (0..self.batch_size).map(|_| dicer.random_range(0..rows.len())).collect();
                let board: Vec<Board> = idx.iter().flat_map(|&i| rows[i].0.symmetries()).collect();
                let head: Vec<f32> = idx.iter().flat_map(|&i| [rows[i].1; 8]).collect();
                let target: Vec<f32> = idx.iter().flat_map(|&i| [rows[i].2; 8]).collect();
                let loss = rater.loss(&board, &head, &target);
                losses.push(loss.to_scalar::<f32>().expect("read scalar loss value"));
                opt.backward_step(&loss).expect("run AdamW backward step on loss");
            }
            callback(round, games, losses);
            round += 1;
        }
    }
}
