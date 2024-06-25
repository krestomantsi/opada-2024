use itertools::{izip, Itertools};

const DIM: usize = 2;

fn main() {
    let x = [-0.1, 0.1, 0.5];
    let lr = 0.01;
    let now = std::time::Instant::now();
    let x_hat = lagrange_multiplier(&x, lr, 10000);
    println!("{:?}", now.elapsed());
    let y = &x_hat[..DIM].try_into().unwrap();
    println!("x_hat is {:?}", x_hat);
    println!("constraint: {:?}", &constraint(y));
    println!("loss: {:?}", &loss(y));
}

fn loss(x: &[f32; DIM]) -> f32 {
    (x[0] + x[1]).powi(2)
}

fn loss_hat(x: &[f32; DIM + 1]) -> f32 {
    (x[0] + x[1]).powi(2) + x[DIM] * constraint(x[0..DIM].try_into().unwrap())
}

fn constraint(x: &[f32; DIM]) -> f32 {
    (x.iter().map(|x| x.powi(2)).sum::<f32>() - 1.0f32).powi(2)
}

fn gradient(x: &[f32; DIM + 1]) -> [f32; DIM + 1] {
    let eps = 1e-4;
    let con = constraint(&[x[0], x[1]]);
    let mut xp = x.clone();
    xp[0] = xp[0] + eps;
    let mut xm = x.clone();
    xm[0] = xm[0] - eps;
    let g1 = (loss_hat(&xp) - loss_hat(&xm)) / (2.0 * eps);
    //
    let mut xp = x.clone();
    xp[1] = xp[1] + eps;
    let mut xm = x.clone();
    xm[1] = xm[1] - eps;
    let g2 = (loss_hat(&xp) - loss_hat(&xm)) / (2.0 * eps);
    let grads = [g1, g2, con];
    grads
}

fn lagrange_multiplier(x: &[f32; DIM + 1], lr: f32, epochs: usize) -> [f32; DIM + 1] {
    //x_hat is the concatenation of x and the lagrange multiplier
    let mut grads;
    let mut opt = adamw_init(x, lr, 0.0, 0.9, 0.999);
    let mut x_hat = x.clone();

    for _ii in 0..epochs {
        grads = gradient(&x_hat);
        x_hat = adamw(&x_hat, &grads, &mut opt);
        //x_hat = sgd(&x_hat, &grads, lr);
        if constraint(&[x_hat[0], x_hat[1]]) < 1e-5 {
            break;
        }
    }
    x_hat
}

#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: f32,
    pub lambda: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub m: [f32; DIM + 1],
    pub v: [f32; DIM + 1],
    pub t: i32,
}

pub fn adamw_init(grads: &[f32], lr: f32, lambda: f32, beta1: f32, beta2: f32) -> Adam {
    let m = grads
        .iter()
        .map(|_| 0.0 as f32)
        .collect::<Vec<f32>>()
        .try_into()
        .unwrap();
    let v = grads
        .iter()
        .map(|_| 0.0 as f32)
        .collect::<Vec<f32>>()
        .try_into()
        .unwrap();
    Adam {
        lr,
        lambda,
        beta1,
        beta2,
        epsilon: 1e-8 as f32,
        m,
        v,
        t: 0,
    }
}

#[inline]
pub fn adamw(x: &[f32; DIM + 1], grads: &[f32; DIM + 1], adam: &mut Adam) -> [f32; DIM + 1] {
    adam.t = adam.t + 1;
    let lr = adam.lr;
    let lambda = adam.lambda;
    let t = adam.t;
    let b: f32 = adam.beta1;
    let b2: f32 = adam.beta2;
    let b11: f32 = 1.0 - b;
    let b22: f32 = 1.0 - b2;
    let m = izip!(adam.m.clone().into_iter(), grads.iter()).map(|(m, grads)| m * b + b11 * grads);
    let v = izip!(adam.v.clone().into_iter(), grads.iter())
        .map(|(v, grads)| v * b2 + b22 * grads * grads);
    let mhat = adam.m.clone().into_iter().map(|x| x / (1.0f32 - b.powi(t)));
    let vhat = adam
        .v
        .clone()
        .into_iter()
        .map(|x| (x / (1.0f32 - b2.powi(t))).sqrt());
    adam.m = m.collect::<Vec<f32>>().try_into().unwrap();
    adam.v = v.collect::<Vec<f32>>().try_into().unwrap();
    izip!(x.iter(), mhat, vhat)
        .map(|(x, mhat, vhat)| x - x * (lambda) - mhat / (vhat + adam.epsilon) * lr)
        .collect::<Vec<f32>>()
        .try_into()
        .unwrap()
}

pub fn sgd(x: &[f32; DIM + 1], grads: &[f32; DIM + 1], lr: f32) -> [f32; DIM + 1] {
    izip!(x, grads)
        .map(|(x, g)| x - lr * g)
        .collect_vec()
        .try_into()
        .unwrap()
}

#[test]
fn gradient_test() {
    let x = [-0.5, 0.2, 0.5];
    let grads = gradient(&x);
    let grads_true = [
        2.0 * (x[0] + x[1]) + x[2] * 4.0 * (x[0].powi(2) + x[1].powi(2) - 1.0) * x[0],
        2.0 * (x[0] + x[1]) + x[2] * 4.0 * (x[0].powi(2) + x[1].powi(2) - 1.0) * x[1],
        constraint(&[x[0], x[1]]),
    ];
    println!("{:?}", grads);
    println!("{:?}", grads_true);
}
