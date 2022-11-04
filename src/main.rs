extern crate ndarray;
extern crate openblas_src;

use ndarray::{arr1, s, Array1, Array2, ArrayView};

pub struct ObjGradHess {
    objective: f64,
    gradient: Array1<f64>,
    hessian: Array2<f64>,
}

impl ObjGradHess {
    pub fn new(obj: f64, grad: Array1<f64>, hess: Array2<f64>) -> Self {
        Self {
            objective: obj,
            gradient: grad,
            hessian: hess,
        }
    }
}

fn rosenbrock(x: &Array1<f64>) -> ObjGradHess {
    // Objective
    let xr = &x.slice(s![1..]);
    let xl = &x.slice(s![..-1]);
    let objective: f64 = (100. * (xr - xl.mapv(|e| e.powf(2.))).map(|e| e.powf(2.))
        + xl.mapv(|e| (-e + 1.).powf(2.)))
    .sum();

    // Gradient
    let len: usize = x.len();
    let xm = &x.slice(s![1..-1]);
    let x2l = &x.slice(s![..-2]);
    let x2r = &x.slice(s![2..]);
    let mut gradient: Array1<f64> = Array1::zeros((len,));
    let mut mslice = gradient.slice_mut(s![1..-1]);
    mslice += &ArrayView::from(
        &(200. * (xm - x2l.mapv(|e| e.powf(2.)))
            - 400. * (x2r - xm.mapv(|e| e.powf(2.))) * xm
            - 2. * (xm.mapv(|e| 1. - e))),
    );
    gradient[0] = -400. * &x[0] * (&x[1] - &x[0].powf(2.)) - 2. * (1. - &x[0]);
    gradient[len - 1] = 200. * (&x[len - 1] - &x[len - 2].powf(2.));

    // Hessian
    // Diagonal +/- 1
    let dpm1 = -400. * xl;
    // Main diagonal
    let mut d0: Array1<f64> = Array1::zeros((len,));
    d0[0] = 1200. * &x[0].powf(2.) - 400. * &x[1] + 2.;
    d0[len - 1] = 200.;
    let mut mdslice = d0.slice_mut(s![1..-1]);
    mdslice += &ArrayView::from(&(202. + 1200. * xm.mapv(|e| e.powf(2.)) - 400. * x2r));

    let mut hessian: Array2<f64> = Array2::from_diag(&d0);
    for (i, e) in dpm1.iter().enumerate() {
        hessian[[i, i + 1]] = *e;
        hessian[[i + 1, i]] = *e;
    }

    ObjGradHess::new(objective, gradient, hessian)
}

fn get_boundaries(z: &Array1<f64>, d: &Array1<f64>, tr: f64) -> [f64; 2] {
    let a: f64 = d.dot(d);
    let b: f64 = 2. * z.dot(d);
    let c: f64 = (z.dot(z)) - tr.powf(2.);
    let aux: f64 = b + ((b * b) - (4. * a * c)).sqrt().copysign(b);
    let ta: f64 = -aux / (2. * a);
    let tb: f64 = -(2. * c) / aux;
    if ta > tb {
        [tb, ta]
    } else {
        [ta, tb]
    }
}

fn model(obj: f64, grad: &Array1<f64>, hess: &Array2<f64>, p: &Array1<f64>) -> f64 {
    obj + p.dot(grad) + 0.5 * p.dot(&(hess.dot(p)))
}

fn trust_ncg(x0: &Array1<f64>) -> Array1<f64> {
    let mut tr: f64 = 1.;
    let max_tr: f64 = 8.;
    let eta: f64 = 0.15;
    let tol: f64 = 1e-8;
    let maxiter: usize = 100;

    let mut x = x0.clone();
    let len = x.len();

    let mut p: Array1<f64>;
    let mut hits_boundary: bool;
    let mut k: usize = 0;
    let objgradhess = rosenbrock(&x);
    let mut obj = objgradhess.objective;
    let mut grad = objgradhess.gradient;
    let mut hess = objgradhess.hessian;
    loop {
        println!("{}", obj);
        let grad_norm: f64 = grad.mapv(|e| e.powf(2.)).sum().sqrt();
        let grad_norm_sqrt: f64 = grad_norm.sqrt();
        let mut epsilon: f64 = grad_norm;
        if grad_norm_sqrt < 0.5 {
            epsilon *= grad_norm_sqrt
        } else {
            epsilon *= 0.5
        };

        let mut z: Array1<f64> = Array1::zeros((len,));
        let mut r = 1. * &grad;
        let mut d = -&grad;

        loop {
            let hd = hess.dot(&d);
            let dhd = d.dot(&hd);

            if dhd <= 0. {
                let [ta, tb] = get_boundaries(&z, &d, tr);
                let pa = &z + ta * &d;
                let pb = &z + tb * &d;
                if model(obj, &grad, &hess, &pa) < model(obj, &grad, &hess, &pb) {
                    p = pa;
                } else {
                    p = pb;
                }
                hits_boundary = false;
                break;
            }

            let r_sq: f64 = r.dot(&r);
            let alpha: f64 = r_sq / dhd;
            let z_next: Array1<f64> = &z + alpha * &d;

            if z_next.mapv(|e| e.powf(2.)).sum().sqrt() >= tr {
                let tb: f64 = get_boundaries(&z, &d, tr)[1];
                hits_boundary = true;
                p = &z + tb * &d;
                break;
            }

            let r_next: Array1<f64> = &r + alpha * &hd;
            let r_next_sq: f64 = r_next.dot(&r_next);

            if r_next_sq.sqrt() < epsilon {
                hits_boundary = false;
                p = z_next;
                break;
            }

            let beta_next: f64 = r_next_sq / r_sq;
            let d_next: Array1<f64> = -&r_next + beta_next * &d;

            z = z_next;
            r = r_next;
            d = d_next;
        }

        let x_prop = &x + &p;
        let objgradhess_prop = rosenbrock(&x_prop);
        let obj_prop = objgradhess_prop.objective;
        let grad_prop = objgradhess_prop.gradient;
        let hess_prop = objgradhess_prop.hessian;

        let actual_reduction = obj - obj_prop;
        let predicted_reduction = obj - model(obj, &grad, &hess, &p);

        if predicted_reduction <= 0. {
            println!("No improvement");
            break;
        }

        let rho = actual_reduction / predicted_reduction;
        if rho < 0.25 {
            tr *= 0.25;
        } else if rho > 0.75 && hits_boundary {
            if 2. * tr <= max_tr {
                tr = 2. * tr;
            } else {
                tr = max_tr;
            }
        }

        if rho > eta {
            x = x_prop;
            obj = obj_prop;
            grad = grad_prop;
            hess = hess_prop;
        }

        k += 1;

        if grad_norm < tol {
            break;
        }

        if k == maxiter {
            break;
        }
    }
    x
}

fn main() {
    let x0: Array1<f64> = arr1(&[1.3, 0.8, 0.7, 1.2]);
    let x = trust_ncg(&x0);
    println!("{}", x);
}
