use clap::Parser;
use rand::Rng;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::io::stdin;
use std::io::stdout;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::thread::sleep;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use z048::Board;
use z048::Dicer;
use z048::Rater;
use z048::Slide;
use z048::Spawn;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    slide_rater: Option<PathBuf>,
    #[arg(long)]
    spawn_rater: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    slide_depth: u8,
    #[arg(long, default_value_t = 4)]
    spawn_depth: u8,
    #[arg(long, default_value_t = 0.0)]
    slide_tau: f64,
    #[arg(long, default_value_t = 0.0)]
    spawn_tau: f64,
    #[arg(long, default_value_t = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock is before the unix epoch").as_nanos() as u64)]
    seed: u64,
    #[arg(long, default_value_t = 80)]
    delay: u64,
}

fn raw(on: bool) {
    let args: &[&str] = if on { &["-echo", "-icanon", "min", "1", "time", "0"] } else { &["sane"] };
    let _ = Command::new("stty").args(args).stderr(Stdio::null()).status();
}

fn read_key() -> char {
    let mut b = [0u8; 1];
    if !matches!(stdin().read(&mut b), Ok(1)) {
        return 'q';
    }
    if b[0] == 0x1b {
        let mut s = [0u8; 2];
        let _ = stdin().read(&mut s);
        return s[1] as char;
    }
    b[0] as char
}

fn draw(board: Board, cursor: Option<(usize, usize)>, rank: u8, header: &str, last: &str, prompt: &str) {
    let grid = <[[u8; 4]; 4]>::from(board);
    let max = grid.into_iter().flatten().max().expect("board always has cells");
    let mut out = String::from("\x1b[H");
    let mut line = |s: String| {
        out.push_str(&s);
        out.push_str("\x1b[K\n");
    };
    line(header.to_string());
    for x in 0..4 {
        let mut row = String::new();
        for y in 0..4 {
            let r = grid[x][y];
            let base = if r == 0 { ".".to_string() } else { (1u64 << r).to_string() };
            let cell = match cursor {
                Some(c) if c == (x, y) => {
                    if r == 0 {
                        format!("[{}]", 1u64 << rank)
                    } else {
                        format!("({base})")
                    }
                }
                _ => base,
            };
            row.push_str(&format!("{cell:>6}"));
        }
        line(row);
    }
    line(format!("score {:.3}  escore {:.0}  max tile {}", board.score(), board.escore(), 1u64 << max));
    if !last.is_empty() {
        line(last.to_string());
    }
    line(prompt.to_string());
    out.push_str("\x1b[J");
    print!("{out}");
    let _ = stdout().flush();
}

fn human_slide(board: Board, header: &str, last: &str) -> Option<Slide> {
    draw(board, None, 0, header, last, "slide to move (human) — arrows to slide, q to quit");
    loop {
        let s = match read_key() {
            'A' => Slide::U,
            'B' => Slide::D,
            'D' => Slide::L,
            'C' => Slide::R,
            'q' => return None,
            _ => continue,
        };
        if board.is_legal_slide(s) {
            return Some(s);
        }
    }
}

fn human_spawn(board: Board, header: &str, last: &str) -> Option<Spawn<4, 2>> {
    let (mut cx, mut cy) = board.empties().next().expect("spawn phase always has an empty cell");
    let mut rank = 1u8;
    loop {
        draw(board, Some((cx, cy)), rank, header, last, "spawn to move (human) — arrows move · [ ]=2/4 · space place · q quit");
        match read_key() {
            'A' => cx = cx.saturating_sub(1),
            'B' => cx = (cx + 1).min(3),
            'D' => cy = cy.saturating_sub(1),
            'C' => cy = (cy + 1).min(3),
            '[' => rank = 1,
            ']' => rank = 2,
            ' ' => {
                let sp = Spawn::<4, 2>::from(((cx, cy), rank));
                if board.is_legal_spawn(sp) {
                    return Some(sp);
                }
            }
            'q' => return None,
            _ => {}
        }
    }
}

fn main() {
    let args = Args::parse();
    let slide_rater: Option<Rater> = args.slide_rater.as_deref().map(|filename| Rater::from(File::open(filename).expect("load checkpoint")));
    let spawn_rater: Option<Rater> = args.spawn_rater.as_deref().map(|filename| Rater::from(File::open(filename).expect("load checkpoint")));
    let slide_who = if slide_rater.is_some() { "AI" } else { "human" };
    let spawn_who = if spawn_rater.is_some() { "AI" } else { "human" };
    let mut dicer = Dicer::from(args.seed);
    let mut board = Board::from(dicer.random::<u64>());

    let human = slide_rater.is_none() || spawn_rater.is_none();
    if human {
        raw(true);
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            raw(false);
            prev(info);
        }));
    }
    let delay = Duration::from_millis(args.delay);
    let header = format!("z048 — slide:{slide_who} spawn:{spawn_who}");
    let mut last = String::new();

    print!("\x1b[2J");
    draw(board, None, 0, &header, &last, "slide to move");
    loop {
        if board.end() || board.iter_legal_slide().next().is_none() {
            break;
        }

        let sl = match &slide_rater {
            Some(r) => r.sample_slide(board, args.slide_depth, args.slide_tau, &mut dicer),
            None => match human_slide(board, &header, &last) {
                Some(s) => s,
                None => break,
            },
        };
        board = board.slide(sl);
        let c = match sl {
            Slide::U => 'U',
            Slide::D => 'D',
            Slide::L => 'L',
            Slide::R => 'R',
        };
        last = format!("last: slide {slide_who} {c}");
        draw(board, None, 0, &header, &last, "spawn to move");
        if slide_rater.is_some() {
            sleep(delay);
        }

        let sp = match &spawn_rater {
            Some(r) => r.sample_spawn(board, args.spawn_depth, args.spawn_tau, &mut dicer),
            None => match human_spawn(board, &header, &last) {
                Some(s) => s,
                None => break,
            },
        };
        board = board.spawn(sp);
        let ((x, y), rk) = sp.cm();
        last = format!("last: spawn {spawn_who} ({x},{y})->{}", 1u64 << rk);
        draw(board, None, 0, &header, &last, "slide to move");
        if spawn_rater.is_some() {
            sleep(delay);
        }
    }

    draw(board, None, 0, &header, &last, "game over");
    if human {
        raw(false);
    }
    println!();
}
