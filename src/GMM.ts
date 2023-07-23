export class GMM {
    constructor(mu: number[], pi: number[], sigma: number[][]) {

    }

    // 過程を逐次見るデバッグ用
    static next_button() {

    }

    // GMMを作るやつ
    static createGMM(data: number[][], init_mu: number[] = GMM.InitMu(data[0].length), init_pi: number[], init_sigma: number[][]): GMM{
        return new GMM(init_mu, init_pi, init_sigma);
    }

    static InitMu(dim_n: number): number[] {
        const mu = new Array<number>(dim_n);
        
        for(var i = 0; i < mu.length; i++){
            mu[i] = GMM.rnorm();
        }

        return mu;
    }

    static InitPi(dim_n: number): number[] {
        const pi = new Array<number>(dim_n).fill(1 / dim_n);
        return pi;
    }

    static InitSigma(dim_n: number): number[][] {
        const sigma = new Array<Array<number>>(dim_n).fill(new Array<number>(dim_n));
        
        for(var i = 0; i < sigma.length; i++){
            sigma[i][i] = 1;
        }        
        
        return sigma;
    }

    // 標準正規分布の乱数
    static rnorm(){
        return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    }
}