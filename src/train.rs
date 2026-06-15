use crate::Board;
use crate::Dicer;
use crate::rater::Rater;
use candle_nn::Optimizer;
use candle_nn::ParamsAdamW;
use rand::Rng;
use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct Train {
    num_round: usize,
    play_games: usize,
    search_depth: u8,
    train_steps: usize,
    batch_size: usize,
    buffer_size: usize,
    random_seed: u64,
    td_lambda: f64,
    tau_a: f64,
    tau_h: f64,
    tau_k: f64,
    adamw_lr: f64,
    adamw_beta1: f64,
    adamw_beta2: f64,
    adamw_eps: f64,
    adamw_wd: f64,
}

impl Default for Train {
    fn default() -> Self {
        Self { num_round: 0, play_games: 64, search_depth: 2, train_steps: 256, batch_size: 256, buffer_size: 1_048_576, random_seed: 0x2048_2048_2048_2048, td_lambda: 0.8, tau_a: 1.0, tau_h: 8.0, tau_k: 0.02, adamw_lr: 1e-3, adamw_beta1: 0.9, adamw_beta2: 0.999, adamw_eps: 1e-8, adamw_wd: 1e-4 }
    }
}

impl Train {
    pub fn train(self, rater: &Rater, callback: impl Fn(usize, Vec<f32>)) {
        let mut dicer = Dicer::from(self.random_seed);
        let mut opt = rater.adamw(ParamsAdamW { lr: self.adamw_lr, beta1: self.adamw_beta1, beta2: self.adamw_beta2, eps: self.adamw_eps, weight_decay: self.adamw_wd });
        let mut rows: Vec<(Board, f32, f32)> = Vec::new();
        let mut next = 0usize;
        for round in 0..self.num_round {
            for _ in 0..self.play_games {
                let mut board = Board::from(dicer.random::<u64>());
                let mut afters: Vec<(Board, f64)> = Vec::new();
                let mut befores: Vec<(Board, f64)> = Vec::new();
                let mut ply = 0usize;
                loop {
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
                        break;
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
            callback(round, losses);
        }
    }
}
