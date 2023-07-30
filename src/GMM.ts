type ArrayOrTexture = number[] | WebGLTexture;

export class GMM<T extends ArrayOrTexture> {
    /* GMM class */
    mu: WebGLTexture;
    pi: WebGLTexture;
    sigma: WebGLTexture;

    constructor(dim_n: number, dist_n: number, mu: T, pi: T, sigma: T) {
        if (typeof mu == 'number') {
            this.mu = mu;
            this.pi = pi;
            this.sigma = sigma;
            // to do
        } else {
            this.mu = mu;
            this.pi = pi;
            this.sigma = sigma;
        }
    }

    /* constructer input check */

    /* CheckArray */
    static CheckArrayMu(dist_n: number, mu: number[]) {
        return dist_n == mu.length;
    }

    static CheckArrayPi(dist_n: number, pi: number[]) {
        return dist_n == pi.length;
    }

    static CheckArraySigma(dist_n: number, sigma: number[]) {
        return dist_n * dist_n == sigma.length;
    }

    /* CheckTexture */
    static CheckTextureMu(dist_n: number, mu: number[]) {
        return dist_n == mu.length;
    }

    static CheckTexturePi(dist_n: number, pi: number[]) {
        return dist_n == pi.length;
    }

    static CheckTextureSigma(dist_n: number, sigma: number[]) {
        return dist_n * dist_n == sigma.length;
    }

    // 過程を逐次見るデバッグ用
    static NextButton() {

    }

    // GMMを作るやつ
    static CreateGMM(
        dim_n: number, dist_n: number, data: number[],
        init_mu: number[] = GMM.InitMu(dist_n), init_pi: number[] = GMM.InitPi(dist_n), init_sigma: number[] = GMM.InitSigma(dim_n, dist_n)
    ): GMM<WebGLTexture> {
        return new GMM<WebGLTexture>(dim_n, dist_n, init_mu, init_pi, init_sigma);
    }

    // 変数を初期化する
    static InitMu(dist_n: number): number[] {
        const mu = new Array<number>(dist_n);

        for (var i = 0; i < dist_n; i++) {
            mu[i] = GMM.Rnorm();
        }

        return mu;
    }

    static InitPi(dist_n: number): number[] {
        const pi = new Array<number>(dist_n).fill(1 / dist_n);
        return pi;
    }

    static InitSigma(dim_n: number, dist_n: number): number[] {
        const sigma = new Array<number>(dim_n * dim_n * dist_n).fill(0);

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