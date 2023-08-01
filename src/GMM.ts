import * as GPGPU from './gpgpu';

export class GMM {
    /* GMM class (3D限定)*/

    static dim_n: number = 3;
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

    static CheckMu(dist_n: number, mu: Float32Array) {
        return dist_n == mu.length;
    }

    static CheckPi(dist_n: number, pi: Float32Array) {
        return dist_n == pi.length;
    }

    static CheckSigma(dist_n: number, sigma: Float32Array) {
        return dist_n * dist_n == sigma.length;
    }

    // 過程を逐次見るデバッグ用
    static NextButton() {

    }

    // GMMを作るやつ
    static CreateGMM(
        dist_n: number, data_n: number, x: Float32Array,
        init_mu: Float32Array = GMM.InitMu(dist_n), init_pi: Float32Array = GMM.InitPi(dist_n), init_sigma: Float32Array = GMM.InitSigma(dist_n)
    ): GMM {
        // Shaders

        const norm_x_shader = `
            uniform vec3 x[${data_n}];

            uniform vec3 mu[${dist_n}];
            uniform mat3 sigma[${dist_n}];

            in float zero;
            out float norm_x;

            void main() {
                int n = gl_VertexID % ${data_n};
                int m = gl_VertexID / ${data_n};

                norm_x = normdist(x[n], mu[m], sigma[m]) + zero;
            }

            float normdist(vec3 x, vec3 mu, mat3 sigma) {
                vec3 nx = x - mu;
                float s2d = nx[0] * (sigma[0][0] * x[0] + sigma[0][1] * x[1] + sigma[0][2] * x[2]) + nx[1] * (sigma[1][0] * x[0] + sigma[1][1] * x[1] + sigma[1][2] * x[2]) + nx[2] * (sigma[2][0] * x[0] + sigma[2][1] * x[1] + sigma[2][2] * x[2]);
                float bottom = sqrt(${(2 * Math.PI) ** data_n} * determinant(sigma));
                float top = exp(-0.5 * s2d);

                return top / bottom;
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

        const mu_pi_sigma_shader = `
            uniform vec3 x[${data_n}];
            uniform vec3 old_mu[${dist_n}];
            uniform sampler2D gamma;
            uniform float gamma_sum[${dist_n}];

            in vec3 zero;
            out vec3 mu;
            out float pi;
            out mat3 sigma;

            void main(){
                int m = gl_VertexID;

                mu = zero;

                for(int k = 0; k < ${data_n}; k++){
                    float gamma_n_m = texelFetch(gamma, ivec2(k, m), 0).r;

                    mu += gamma_n_m * x[k];
                    sigma += gamma_n_m * mat3(
                        (x[k][0] - old_mu[m][0]) * (x[k][0] - old_mu[m][0]), (x[k][1] - old_mu[m][1]) * (x[k][0] - old_mu[m][0]), (x[k][2] - old_mu[m][2]) * (x[k][0] - old_mu[m][0]),
                        (x[k][0] - old_mu[m][0]) * (x[k][1] - old_mu[m][1]), (x[k][1] - old_mu[m][1]) * (x[k][1] - old_mu[m][1]), (x[k][2] - old_mu[m][2]) * (x[k][1] - old_mu[m][1]),
                        (x[k][0] - old_mu[m][0]) * (x[k][2] - old_mu[m][2]), (x[k][1] - old_mu[m][1]) * (x[k][2] - old_mu[m][2]), (x[k][2] - old_mu[m][2]) * (x[k][2] - old_mu[m][2])
                    );
                }

                mu /= gamma_sum[m];
                sigma /= gamma_sum[m];

                pi = gamma_sum[m] / ${data_n};
            }
        `

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
        
        const norm_x = new Float32Array(data_n * dist_n);
        const norm_x_sum = new Float32Array(data_n);

        var log_p;

        const gamma = new Float32Array(data_n * dist_n);
        const gamma_sum = new Float32Array(dist_n);

        const pi = init_pi;
        const mu = init_mu;
        const sigma = init_sigma;

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
                'norm_x': gpgpu.makeTextureInfo('float', [data_n, dist_n], norm_x),
                'norm_x_sum': norm_x_sum,
                'pi': pi
            }
        };

        const gamma_param = {
            id: 'gamma_shader',
            vertexShader: gamma_shader,
            args: {
                'zero': data_n_dist_n_zero,
                'norm_x': gpgpu.makeTextureInfo('float', [data_n, dist_n], norm_x),
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
                'gamma': gpgpu.makeTextureInfo('float', [data_n, dist_n], gamma),
                'gamma_sum': gamma_sum
            }
        }

        const mu_pi_sigma_param = {
            id: 'mu_pi_sigma_shader',
            vertexShader: mu_pi_sigma_shader,
            args: {
                'zero': dist_n_zero,
                'x': x,
                'old_mu': mu,
                'gamma': gpgpu.makeTextureInfo('float', [data_n, dist_n], gamma),
                'gamma_sum': gamma_sum,
                'mu': mu,
                'pi': pi,
                'sigma': sigma
            }
        }

        for(let i = 0; i < 100; i++) {
            // to do
            // 1. norm_x, norm_sum

            gpgpu.compute(norm_x_param);
            gpgpu.compute(norm_x_sum_param);

            // 2. log_p, judge break
            
            log_p = log_p_func(norm_x_sum);

            // 3. gamma


            // 4. gamma_sum
            // 5. mu, pi, sigma
        }

        gpgpu.clear(norm_x_param.id);
        gpgpu.clear(norm_x_sum_param.id);

        return new GMM(dist_n, init_mu, init_pi, init_sigma);

        /*
        const norm_x_func = (gpgpu: any, norm_x_shader: string, norm_x_sum_shader: string, x: Float32Array, mu: Float32Array, pi: Float32Array, sigma: Float32Array) => {
            const data_n_zero = new Float32Array(data_n);
            const norm_x = new Float32Array(data_n * dist_n);
            
            const norm_x_param = {
                id: 'log_p_func_norm_x',
                vertexShader: norm_x_shader,
                args: {
                    'zero': data_n_zero,
                    'norm_x': norm_x,
                    'x': x,
                    'mu': mu,
                    'sigma': sigma
                }
            };

            gpgpu.compute(norm_x_param);
            gpgpu.clear(norm_x_param.id);

            const norm_x_sum = new Float32Array(data_n);
            
            const norm_x_sum_param = {
                id: 'log_p_func_norm_x_sum',
                vertexShader: norm_x_sum_shader,
                args: {
                    'zero': data_n_zero,
                    'norm_x': gpgpu.makeTextureInfo('float', norm_x, [data_n, dist_n]),
                    'norm_x_sum': norm_x_sum,
                    'pi': pi
                }
            };

            gpgpu.compute(norm_x_sum_param);
            gpgpu.clear(norm_x_sum_param.id);

            return [norm_x, norm_x_sum];
        };
        */
    }

    // 変数を初期化する
    static InitMu(dist_n: number): Float32Array {
        const mu = new Array<number>(dist_n);

        for (var i = 0; i < dist_n; i++) {
            mu[i] = GMM.Rnorm();
        }

        return new Float32Array(mu);
    }

    static InitPi(dist_n: number): Float32Array {
        const pi = new Array<number>(dist_n).fill(1.0 / dist_n);
        return new Float32Array(pi);
    }

    static InitSigma(dist_n: number): Float32Array {
        const sigma = new Array<number>(9 * dist_n).fill(0.0);

        for (var j = 0; j < dist_n; j++) {
            sigma[0 + j * 9] = 1.0;
            sigma[4 + j * 9] = 1.0;
            sigma[8 + j * 9] = 1.0;
        }

        return new Float32Array(sigma);
    }

    // 標準正規分布の乱数
    static Rnorm() {
        return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    }
}