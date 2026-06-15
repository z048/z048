use crate::Board;
use crate::Dicer;
use crate::Slide;
use crate::Spawn;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_core::Var;
use candle_nn::Activation;
use candle_nn::AdamW;
use candle_nn::Linear;
use candle_nn::Module;
use candle_nn::Optimizer;
use candle_nn::ParamsAdamW;
use candle_nn::Sequential;
use candle_nn::VarMap;
use candle_nn::seq;
use rand::Rng;
use std::fs::File;
use std::fs::write;
use std::io::Read;
use std::iter::once;
use std::path::Path;

pub struct Rater {
    varmap: VarMap,
    model: Sequential,
    device: Device,
}

impl Rater {
    const RANKS: usize = 16;
    const MRANK: usize = Self::RANKS - 1;
    const INPUT: usize = 4 * 4 * Self::RANKS;
    pub fn boards(&self, boards: &[Board]) -> Tensor {
        let mut x = vec![0f32; boards.len() * Self::INPUT];
        for (i, &b) in boards.iter().enumerate() {
            for (cell, r) in <[[u8; 4]; 4]>::from(b).into_iter().flatten().enumerate() {
                x[i * Self::INPUT + cell * Self::RANKS + (r as usize).min(Self::MRANK)] = 1.0;
            }
        }
        Tensor::from_vec(x, (boards.len(), Self::INPUT), &self.device).expect("build one-hot board batch tensor")
    }

    pub fn forward(&self, boards: &[Board]) -> Tensor {
        self.model.forward(&self.boards(boards)).expect("run forward pass through rater model")
    }

    pub fn loss(&self, boards: &[Board], head: &[f32], target: &[f32]) -> Tensor {
        let out = self.forward(boards);
        let head = Tensor::from_slice(head, head.len(), &self.device).expect("build head-selector tensor");
        let target = Tensor::from_slice(target, target.len(), &self.device).expect("build target tensor");
        let mask = Tensor::stack(&[head.affine(-1.0, 1.0).expect("compute 1 - head"), head.clone()], 1).expect("stack head one-hot mask");
        let pred = (out * mask).expect("apply head mask to outputs").sum(1).expect("sum masked outputs over heads");
        (pred - target).expect("compute prediction residual").sqr().expect("square residual").mean(0).expect("mean squared error over batch")
    }

    pub fn sample_slide(&self, board: Board, depth: u8, tau: f64, dicer: &mut Dicer) -> Slide {
        let phi = board.score();
        dicer.softmax(
            board
                .iter_legal_slide()
                .map(|sl| {
                    let child = board.slide(sl);
                    (sl, (child.score() - phi) + self.minimize(child, depth, (f64::NEG_INFINITY, f64::INFINITY)))
                })
                .collect(),
            tau,
        )
    }

    pub fn sample_spawn(&self, board: Board, depth: u8, tau: f64, dicer: &mut Dicer) -> Spawn<4, 2> {
        let phi = board.score();
        dicer.softmax(
            board
                .iter_legal_spawn()
                .map(|sp| {
                    let child = board.spawn(sp);
                    let dphi = child.score() - phi;
                    (sp, -if child.end() { dphi } else { dphi + self.maximize(child, depth, (f64::NEG_INFINITY, f64::INFINITY)) })
                })
                .collect(),
            tau,
        )
    }

    pub fn adamw(&self, config: ParamsAdamW) -> AdamW {
        AdamW::new(self.varmap.all_vars(), config).expect("build AdamW optimizer over varmap")
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) {
        let mut layers = vec![];
        {
            let map = self.varmap.data().lock().expect("lock varmap data");
            let mut i = 0;
            while let Some(w) = map.get(&format!("{i}.weight")) {
                let b = map.get(&format!("{i}.bias")).unwrap_or_else(|| panic!("missing {i}.bias"));
                let weight = w.as_tensor().flatten_all().expect("flatten weight").to_vec1::<f32>().expect("read weight values");
                let bias = b.as_tensor().flatten_all().expect("flatten bias").to_vec1::<f32>().expect("read bias values");
                layers.push((weight, bias));
                i += 1;
            }
        }
        let bytes = postcard::to_allocvec(&layers).expect("serialize layers to postcard");
        write(path, bytes).expect("write postcard layers");
    }

    fn minimize(&self, board: Board, depth: u8, (alpha, mut beta): (f64, f64)) -> f64 {
        if depth == 0 {
            return self.forward(&[board]).to_vec2::<f32>().expect("read v_after from leaf eval")[0][0] as f64;
        }
        let phi = board.score();
        let mut best = f64::INFINITY;
        for sp in board.iter_legal_spawn() {
            let child = board.spawn(sp);
            let dphi = child.score() - phi;
            best = best.min(if child.end() { dphi } else { dphi + self.maximize(child, depth - 1, (alpha - dphi, beta - dphi)) });
            beta = beta.min(best);
            if beta <= alpha {
                break;
            }
        }
        best
    }

    fn maximize(&self, board: Board, depth: u8, (mut alpha, beta): (f64, f64)) -> f64 {
        if depth == 0 {
            return self.forward(&[board]).to_vec2::<f32>().expect("read v_before from leaf eval")[0][1] as f64;
        }
        let phi = board.score();
        let mut best = f64::NEG_INFINITY;
        for sl in board.iter_legal_slide() {
            let child = board.slide(sl);
            let dphi = child.score() - phi;
            best = best.max(dphi + self.minimize(child, depth - 1, (alpha - dphi, beta - dphi)));
            alpha = alpha.max(best);
            if beta <= alpha {
                break;
            }
        }
        best
    }
}

impl From<File> for Rater {
    fn from(mut value: File) -> Self {
        let mut bytes = Vec::new();
        value.read_to_end(&mut bytes).expect("read checkpoint file");
        let layers: Vec<(Vec<f32>, Vec<f32>)> = postcard::from_bytes(&bytes).expect("deserialize postcard layers");
        layers.into()
    }
}

impl From<Vec<(Vec<f32>, Vec<f32>)>> for Rater {
    fn from(value: Vec<(Vec<f32>, Vec<f32>)>) -> Self {
        let varmap = VarMap::new();
        {
            let mut map = varmap.data().lock().expect("lock varmap data");
            for (i, (weight, bias)) in value.into_iter().enumerate() {
                let out = bias.len();
                let inp = weight.len() / out;
                map.insert(format!("{i}.weight"), Var::from_tensor(&Tensor::from_vec(weight, (inp, out), &Device::Cpu).expect("build weight tensor")).expect("wrap weight tensor as var"));
                map.insert(format!("{i}.bias"), Var::from_tensor(&Tensor::from_vec(bias, out, &Device::Cpu).expect("build bias tensor")).expect("wrap bias tensor as var"));
            }
        }
        (varmap, Device::Cpu).into()
    }
}

impl From<(Vec<usize>, u64)> for Rater {
    fn from((hidden, seed): (Vec<usize>, u64)) -> Self {
        (hidden, seed, Device::Cpu).into()
    }
}

impl From<(Vec<usize>, u64, Device)> for Rater {
    fn from((hidden, seed, device): (Vec<usize>, u64, Device)) -> Self {
        let d: Vec<usize> = once(Self::INPUT).chain(hidden).chain(once(2)).collect();
        let varmap = VarMap::new();
        let mut dicer = Dicer::from(seed);
        {
            let mut map = varmap.data().lock().expect("lock varmap data for init");
            for i in 0..d.len() - 1 {
                let w: Vec<f32> = if i + 2 < d.len() {
                    let lim = if i == 0 { (6.0f32 / 16.0).sqrt() } else { (6.0f32 / d[i] as f32).sqrt() };
                    (0..d[i] * d[i + 1]).map(|_| dicer.random_range(-lim..lim)).collect()
                } else {
                    vec![0.0; d[i] * d[i + 1]]
                };
                map.insert(format!("{i}.weight"), Var::from_tensor(&Tensor::from_vec(w, (d[i], d[i + 1]), &device).expect("build weight tensor")).expect("wrap weight tensor as var"));
                map.insert(format!("{i}.bias"), Var::from_tensor(&Tensor::zeros(d[i + 1], DType::F32, &device).expect("build bias tensor")).expect("wrap bias tensor as var"));
            }
        }
        Rater::from((varmap, device))
    }
}

impl From<(VarMap, Device)> for Rater {
    fn from((varmap, device): (VarMap, Device)) -> Self {
        let mut model = seq();
        {
            let map = varmap.data().lock().expect("lock varmap data for wiring");
            let mut i = 0;
            let mut prev_out = Self::INPUT;
            while let Some(w) = map.get(&format!("{i}.weight")) {
                let (inp, out) = w.dims2().expect("layer weight must be a 2-D [in, out] tensor");
                assert_eq!(inp, prev_out, "layer {i} input width does not chain");
                let b = map.get(&format!("{i}.bias")).unwrap_or_else(|| panic!("missing {i}.bias"));
                if i > 0 {
                    model = model.add(Activation::Relu);
                }
                model = model.add(Linear::new(w.as_tensor().t().expect("transpose weight to a [out, in] view"), Some(b.as_tensor().clone())));
                prev_out = out;
                i += 1;
            }
            assert!(i >= 1, "varmap records no layers");
            assert_eq!(prev_out, 2, "head must have 2 outputs");
            assert_eq!(map.len(), 2 * i, "varmap holds tensors that are not part of the layer chain");
        }
        Rater { varmap, model, device }
    }
}
