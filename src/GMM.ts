import * as GPGPU from './gpgpu';
import { kmeansInc } from './KmeansInc';

export class GMM {
    /* GMM class (3D限定)*/

    dist_n: number;
    mu: Float32Array;
    pi: Float32Array;
    sigma: Float32Array;

    constructor(dist_n: number, mu: Float32Array, pi: Float32Array, sigma: Float32Array) {
        this.dist_n = dist_n;

        this.mu = mu;
        this.pi = pi;
        this.sigma = sigma;
    }

    /* constructer input check */

    static CheckMu(dist_n: number, mu: Float32Array): boolean {
        return dist_n == mu.length;
    }

    static CheckPi(dist_n: number, pi: Float32Array): boolean {
        return dist_n == pi.length;
    }

    static CheckSigma(dist_n: number, sigma: Float32Array): boolean {
        return dist_n * dist_n == sigma.length;
    }

    // GMMを作るやつ
    static CreateGMM(
        dist_n: number, data_n: number, x: Float32Array, regularation_value: number = 0.0, max_iteration_time: number = 100, log_p_tolerance_value: number = 1e-6
    ): {GMM: GMM, gamma: Float32Array, gamma_sum: Float32Array} {

        // Initialize
        const x_mu_var = this.EvalXMuVar(data_n, x);
        const x_var = x_mu_var[1];

        const init_mu = this.InitMu(dist_n);
        const init_pi = this.InitPi(dist_n);
        const init_sigma = this.InitSigma(dist_n, x_var);

        // Shaders
        // const inv_sigma_shader = 

        const norm_x_shader = `
            uniform vec3 x[${data_n}];

            uniform vec3 mu[${dist_n}];
            uniform mat3 sigma[${dist_n}];

            in float zero;
            out float norm_x;

            float normdist(vec3 x, vec3 mu, mat3 sigma) {
                vec3 nx = x - mu;
                mat3 invsigma = inverse(sigma); // ここ分けてもよさそう
                
                float s2d = nx[0] * (
                        invsigma[0][0] * nx[0]
                         + invsigma[0][1] * nx[1]
                         + invsigma[0][2] * nx[2]
                    ) + nx[1] * (
                    invsigma[1][0] * nx[0]
                         + invsigma[1][1] * nx[1]
                         + invsigma[1][2] * nx[2]
                    ) + nx[2] * (
                    invsigma[2][0] * nx[0]
                         + invsigma[2][1] * nx[1]
                         + invsigma[2][2] * nx[2]
                    ); // 二次形式
                float bottom = sqrt(${(2 * Math.PI) ** data_n} * determinant(sigma));
                float top = exp(-0.5 * s2d);

                return top / bottom;
            }

            void main() {
                int n = gl_VertexID % ${data_n};
                int m = gl_VertexID / ${data_n};

                norm_x = normdist(x[n], mu[m], sigma[m]) + zero;
            }
        `

        const norm_x_sum_shader = `
            uniform sampler2D norm_x;
            uniform float pi[${dist_n}];

            in float zero;
            out float norm_x_sum;

            void main() {
                int n = gl_VertexID;
                norm_x_sum = zero;

                for(int k = 0; k < ${dist_n}; k++){
                    norm_x_sum += pi[k] * texelFetch(norm_x, ivec2(n, k), 0).r; // 列、行の順序で指定
                }
            }
        `

        const gamma_shader = `
            uniform sampler2D norm_x;
            uniform float norm_x_sum[${data_n}];

            uniform float pi[${dist_n}];

            in float zero;
            out float gamma;

            void main() {
                int n = gl_VertexID % ${data_n}; // 連続していると早くなるのでうれしい
                int m = gl_VertexID / ${data_n};

                gamma = pi[m] * texelFetch(norm_x, ivec2(n, m), 0).r / norm_x_sum[n] + zero; // 列、行の順序で指定
            }
        `;

        const gamma_sum_shader = `
            uniform sampler2D gamma;

            in float zero;
            out float gamma_sum;

            void main() {
                int m = gl_VertexID;

                gamma_sum = zero;

                for(int k = 0; k < ${data_n}; k++){
                    gamma_sum += texelFetch(gamma, ivec2(k, m), 0).r;
                }
            }
        `

        const sigma_shader = `
            uniform vec3 x[${data_n}];
            uniform vec3 mu[${dist_n}];
            uniform sampler2D gamma;
            uniform float gamma_sum[${dist_n}];
            
            in float zero;
            out vec3 sigma;

            void main() {
                int m = gl_VertexID / 3;
                int k = gl_VertexID % 3;
                
                sigma = vec3(zero);

                for(int n = 0; n < ${data_n}; n++){
                    float gamma_n_m = texelFetch(gamma, ivec2(n, m), 0).r;

                    sigma += gamma_n_m * vec3(
                        (x[n][k] - mu[m][k]) * (x[n][0] - mu[m][0]),
                        (x[n][k] - mu[m][k]) * (x[n][1] - mu[m][1]),
                        (x[n][k] - mu[m][k]) * (x[n][2] - mu[m][2])
                    );
                }

                sigma /= gamma_sum[m];
                
                // regularation_value倍した単位行列を加算
                sigma[k] += ${regularation_value.toFixed(10)};
            }
        `;

        // outにmat3を指定することは許されなかったのでsigmaは別に計算……
        const mu_pi_shader = `
            uniform vec3 x[${data_n}];
            uniform sampler2D gamma;
            uniform float gamma_sum[${dist_n}];

            in float zero;
            out vec3 mu;
            out float pi;

            void main() {
                int m = gl_VertexID;

                mu = vec3(0.0);

                for(int n = 0; n < ${data_n}; n++){
                    float gamma_n_m = texelFetch(gamma, ivec2(n, m), 0).r;

                    mu += gamma_n_m * x[n];
                }

                mu /= gamma_sum[m];
                pi = gamma_sum[m] / ${data_n}.0 + zero;
            }
        `;

        // Functions

        const log_p_func = (norm_x_sum: Float32Array) => {
            var log_p = 0;

            for(let k = 0; k < data_n; k++){
                log_p += Math.log(norm_x_sum[k]);
            }

            return log_p;
        };

        // Variables
        const gpgpu = GPGPU.CreateGPGPU();

        const data_n_dist_n_zero = new Float32Array(data_n * dist_n);
        const data_n_zero = new Float32Array(data_n);
        const dist_n_zero = new Float32Array(dist_n);
        const dist_n_vec3_zero = new Float32Array(dist_n * 3);

        const norm_x = new Float32Array(data_n * dist_n);
        const norm_x_sum = new Float32Array(data_n);

        var log_p;

        const gamma = new Float32Array(data_n * dist_n);
        const gamma_sum = new Float32Array(dist_n);

        const mu = init_mu.slice();
        const pi = init_pi.slice();
        const sigma = init_sigma.slice();

        // Parameters

        const norm_x_param = {
            id: 'norm_x_shader',
            vertexShader: norm_x_shader,
            args: {
                'zero': data_n_dist_n_zero,
                'norm_x': norm_x,
                'x': x,
                'mu': mu,
                'sigma': sigma
            }
        };

        const norm_x_sum_param = {
            id: 'norm_x_sum_shader',
            vertexShader: norm_x_sum_shader,
            args: {
                'zero': data_n_zero,
                'norm_x': gpgpu.makeTextureInfo('float', [dist_n, data_n], norm_x),
                'norm_x_sum': norm_x_sum,
                'pi': pi
            }
        };

        const gamma_param = {
            id: 'gamma_shader',
            vertexShader: gamma_shader,
            args: {
                'zero': data_n_dist_n_zero,
                'norm_x': gpgpu.makeTextureInfo('float', [dist_n, data_n], norm_x),
                'norm_x_sum': norm_x_sum,
                'pi': pi,
                'gamma': gamma
            }
        }

        const gamma_sum_param = {
            id: 'gamma_sum_shader',
            vertexShader: gamma_sum_shader,
            args: {
                'zero': dist_n_zero,
                'gamma': gpgpu.makeTextureInfo('float', [dist_n, data_n], gamma),
                'gamma_sum': gamma_sum
            }
        }

        const sigma_param = {
            id: 'sigma_shader',
            vertexShader: sigma_shader,
            args: {
                'x': x,
                'mu': mu,
                'gamma': gpgpu.makeTextureInfo('float', [dist_n, data_n], gamma),
                'gamma_sum': gamma_sum,
                'zero': dist_n_vec3_zero,
                'sigma': sigma
            }
        }

        const mu_pi_param = {
            id: 'mu_pi_shader',
            vertexShader: mu_pi_shader,
            args: {
                'x': x,
                'gamma': gpgpu.makeTextureInfo('float', [dist_n, data_n], gamma),
                'gamma_sum': gamma_sum,
                'zero': dist_n_zero,
                'mu': mu,
                'pi': pi
            }
        }

        // 1. norm_x, norm_sum

        gpgpu.compute(norm_x_param);
        gpgpu.compute(norm_x_sum_param);
        
        for(var i = 0; i < max_iteration_time; i++) {
            // 2. gamma, gamma_sum

            gpgpu.compute(gamma_param);
            gpgpu.compute(gamma_sum_param);

            // 3. mu, pi, sigma

            gpgpu.compute(sigma_param);
            gpgpu.compute(mu_pi_param);

            // 4. norm_x, norm_sum

            gpgpu.compute(norm_x_param);
            gpgpu.compute(norm_x_sum_param);

            // 5. log_p, judge break
            
            log_p = log_p_func(norm_x_sum);

            if (log_p < log_p_tolerance_value) {
                break;
            }
        }

        gpgpu.clear(norm_x_param.id);
        gpgpu.clear(norm_x_sum_param.id);

        if (i > 0) {
            gpgpu.clear(gamma_param.id);
            gpgpu.clear(gamma_sum_param.id);
            gpgpu.clear(sigma_param);
            gpgpu.clear(mu_pi_param);
        }

        return {
            GMM: new GMM(dist_n, mu, pi, sigma),
            gamma: gamma,
            gamma_sum: gamma_sum
        };
    }

    // データ点全体の平均と標準偏差を求める
    static EvalXMuVar(data_n: number, x: Float32Array): Float32Array[] {
        const gpgpu = GPGPU.CreateGPGPU();

        const zero = new Float32Array(3);
        const x_mu = new Float32Array(3);
        const x_var = new Float32Array(3);
        
        const x_mu_var_shader = `
            uniform vec3 x[${data_n}];

            in vec3 zero;
            out vec3 x_mu;
            out vec3 x_var;

            void main() {
                x_mu = zero;

                for (int k = 0; k < ${data_n}; k++) {
                    x_mu += x[k];
                }

                x_mu /= ${data_n}.0;

                x_var = zero;

                for (int k = 0; k < ${data_n}; k++) {
                    vec3 nx = (x[k] - x_mu);
                    x_var += nx * nx;
                }

                x_var /= ${data_n}.0;
                x_var = x_var;
            }
        `

        const x_mu_var_param = {
            id: 'x_mu_var_shader',
            vertexShader: x_mu_var_shader,
            args: {
                'x': x,
                'zero': zero,
                'x_mu': x_mu,
                'x_var': x_var
            }
        }

        gpgpu.compute(x_mu_var_param);
        gpgpu.clear(x_mu_var_param.id);
        
        return [x_mu, x_var];
    }

    //　gammaから事後確率を計算
    static PostProbByGamma(dist_n: number, data_n: number, gamma: Float32Array): Float32Array {
        // Shaders
        const gamma_sum_shader = `
            uniform sampler2D gamma;
            
            in float zero;
            out float gamma_sum;

            void main() {
                int n = gl_VertexID;
                gamma_sum = zero;

                for (int m = 0; m < ${dist_n}; m++) {
                    gamma_sum += texelFetch(gamma, ivec2(n, m), 0).r;
                }
            }
        `

        const post_prob_shader = `
            uniform float gamma_sum[${data_n}];

            in float gamma;
            out float post_prob;

            void main() {
                int n = gl_VertexID % ${data_n};
                
                post_prob = gamma / gamma_sum[n];
            }
        `;

        // Variables
        const gpgpu = GPGPU.CreateGPGPU();
        const data_n_zero =  new Float32Array(data_n);
        const gamma_sum = new Float32Array(data_n);
        const post_prob = new Float32Array(dist_n * data_n);
        
        // Parameters
        const gamma_sum_param = {
            id: 'gamma_sum_shader',
            vertexShader: gamma_sum_shader,
            args: {
                'gamma': gpgpu.makeTextureInfo('float', [dist_n, data_n], gamma),
                'zero': data_n_zero,
                'gamma_sum': gamma_sum
            }
        }

        const post_prob_param = {
            id: 'post_prob_shader',
            vertexShader: post_prob_shader,
            args: {
                'gamma_sum': gamma_sum,
                'gamma': gamma,
                'post_prob': post_prob
            }
        }

        gpgpu.compute(gamma_sum_param);
        gpgpu.clear(gamma_sum_param.id);

        gpgpu.compute(post_prob_param);
        gpgpu.clear(post_prob_param.id);

        return post_prob;
    }

    static ClusteringByPostProb(dist_n: number, data_n: number, post_prob: Float32Array): Float32Array {
        // Shaders
        const clustering_shader = `
            uniform sampler2D post_prob;

            in float zero;
            out float x_cluster;

            void main() {
                int n = gl_VertexID;
                
                float max_post_prob = zero;

                for (int m = 0; m < ${data_n}; m++) {
                    float post_prob_n_m = texelFetch(post_prob, ivec2(n, m), 0).r;
                    
                    if (post_prob_n_m > max_post_prob) {
                        max_post_prob = post_prob_n_m;
                        x_cluster = m;
                    }
                }
            }
        `;

        // Variables
        const gpgpu = GPGPU.CreateGPGPU();
        const data_n_zero = new Float32Array(data_n);
        const x_cluster = new Float32Array(data_n);

        // Parameters
        const clustering_param = {
            id: 'clustering_shader',
            vertexShader: clustering_shader,
            args: {
                'post_prob': gpgpu.makeTextureInfo('float', [dist_n, data_n], post_prob),
                'zero': data_n_zero,
                'x_cluster': x_cluster
            }
        }

        gpgpu.compute(clustering_param);
        gpgpu.clear(clustering_param.id);

        return x_cluster;
    }

    // 変数を初期化する
    static InitMu(dist_n: number): Float32Array {
        const mu_x_cluster = kmeansInc(dist_n, data_n, x);
        const mu = mu_x_cluster[0];

        /*
        // あんまりよくない初期値
        for (let i = 0; i< dist_n; i++) {
            let norm_value = this.Rnorm();
            mu[0 + i * 3] = x_var[0] * norm_value + x_mu[0];
            mu[1 + i * 3] = x_var[1] * norm_value + x_mu[1];
            mu[2 + i * 3] = x_var[1] * norm_value + x_mu[2];
        }
        */

        return mu;
    }

    static InitPi(dist_n: number): Float32Array {
        return new Float32Array(dist_n).fill(1.0 / dist_n);
    }

    static InitSigma(dist_n: number, x_var: Float32Array): Float32Array {
        const sigma = new Float32Array(9 * dist_n).fill(0.0);

        for (let i = 0; i < dist_n; i++) {
            sigma[0 + i * 9] = x_var[0];
            sigma[4 + i * 9] = x_var[1];
            sigma[8 + i * 9] = x_var[2];
        }

        return sigma;
    }

    // 標準正規分布の乱数(Box-Muller法)
    static Rnorm() {
        return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    }
}