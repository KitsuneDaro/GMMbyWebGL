"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.GMM = void 0;
class GMM {
    constructor(dim_n, dist_n, mu, pi, sigma) {
        if (typeof mu == 'number') {
            this.mu = mu;
            this.pi = pi;
            this.sigma = sigma;
            // to do
        }
        else {
            this.mu = mu;
            this.pi = pi;
            this.sigma = sigma;
        }
    }
    /* constructer input check */
    /* CheckArray */
    static CheckArrayMu(dist_n, mu) {
        return dist_n == mu.length;
    }
    static CheckArrayPi(dist_n, pi) {
        return dist_n == pi.length;
    }
    static CheckArraySigma(dist_n, sigma) {
        return dist_n * dist_n == sigma.length;
    }
    /* CheckTexture */
    static CheckTextureMu(dist_n, mu) {
        return dist_n == mu.length;
    }
    static CheckTexturePi(dist_n, pi) {
        return dist_n == pi.length;
    }
    static CheckTextureSigma(dist_n, sigma) {
        return dist_n * dist_n == sigma.length;
    }
    // 過程を逐次見るデバッグ用
    static NextButton() {
    }
    // GMMを作るやつ
    static CreateGMM(dim_n, dist_n, data, init_mu = GMM.InitMu(dist_n), init_pi = GMM.InitPi(dist_n), init_sigma = GMM.InitSigma(dim_n, dist_n)) {
        return new GMM(dim_n, dist_n, init_mu, init_pi, init_sigma);
    }
    // 変数を初期化する
    static InitMu(dist_n) {
        const mu = new Array(dist_n);
        for (var i = 0; i < dist_n; i++) {
            mu[i] = GMM.Rnorm();
        }
        return mu;
    }
    static InitPi(dist_n) {
        const pi = new Array(dist_n).fill(1 / dist_n);
        return pi;
    }
    static InitSigma(dim_n, dist_n) {
        const sigma = new Array(dim_n * dim_n * dist_n).fill(0);
        for (var j = 0; j < dist_n; j++) {
            for (var i = 0; i < dim_n; i++) {
                sigma[i + i * dim_n + j * dim_n * dim_n] = 1;
            }
        }
        return sigma;
    }
    // 標準正規分布の乱数
    static Rnorm() {
        return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    }
}
exports.GMM = GMM;
