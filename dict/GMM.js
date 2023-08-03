"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.GMM = void 0;
const GPGPU = __importStar(require("./gpgpu"));
const KmeansInc_1 = require("./KmeansInc");
class GMM {
    constructor(dist_n, mu, pi, sigma) {
        this.dist_n = dist_n;
        this.mu = mu;
        this.pi = pi;
        this.sigma = sigma;
    }
    /* constructer input check */
    static CheckMu(dist_n, mu) {
        return dist_n == mu.length;
    }
    static CheckPi(dist_n, pi) {
        return dist_n == pi.length;
    }
    static CheckSigma(dist_n, sigma) {
        return dist_n * dist_n == sigma.length;
    }
    // 過程を逐次見るデバッグ用
    static NextButton() {
    }
    // GMMを作るやつ
    static CreateGMM(dist_n, data_n, x, regularation_value = 0.0, log_p_tolerance_value = 1e-6) {
        // Initialize
        const x_mu_std = this.EvalXMuStd(dist_n, data_n, x);
        const x_mu = x_mu_std[0];
        const x_std = x_mu_std[1];
        const init_mu = this.InitMu(dist_n, x_mu, x_std);
        const init_pi = this.InitPi(dist_n);
        const init_sigma = this.InitSigma(dist_n, x_std);
        // Shaders
        // const inv_sigma_shader = 
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
                mat3 invsigma = inverse(sigma); // ここ分けてもよさそう
                
                float s2d = nx[0] * (invsigma[0][0] * nx[0] + invsigma[0][1] * nx[1] + invsigma[0][2] * nx[2]) + nx[1] * (invsigma[1][0] * nx[0] + invsigma[1][1] * nx[1] + invsigma[1][2] * nx[2]) + nx[2] * (invsigma[2][0] * nx[0] + invsigma[2][1] * nx[1] + invsigma[2][2] * nx[2]);
                float bottom = sqrt(${(2 * Math.PI) ** data_n} * determinant(sigma));
                float top = exp(-0.5 * s2d);

                return top / bottom;
            }
        `;
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
        `;
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
        `;
        const mu_pi_sigma_shader = `
            uniform vec3 x[${data_n}];
            uniform vec3 old_mu[${dist_n}];
            uniform sampler2D gamma;
            uniform float gamma_sum[${dist_n}];

            in float zero;
            out vec3 mu;
            out float pi;
            out mat3 sigma;

            void main(){
                int m = gl_VertexID;

                mu = vec3(0.0);
                sigma = mat3(0.0);

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
                sigma += mat3(${regularation_value}); // regularation_value倍した単位行列

                pi = gamma_sum[m] / ${data_n} + zero;
            }
        `;
        // Functions
        const log_p_func = (norm_x_sum) => {
            var log_p = 0;
            for (let k = 0; k < data_n; k++) {
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
        };
        const gamma_sum_param = {
            id: 'gamma_sum_shader',
            vertexShader: gamma_sum_shader,
            args: {
                'zero': dist_n_zero,
                'gamma': gpgpu.makeTextureInfo('float', [data_n, dist_n], gamma),
                'gamma_sum': gamma_sum
            }
        };
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
        };
        for (let i = 0; i < 100; i++) {
            // to do
            // 1. norm_x, norm_sum
            gpgpu.compute(norm_x_param);
            gpgpu.compute(norm_x_sum_param);
            // 2. log_p, judge break
            log_p = log_p_func(norm_x_sum);
            if (log_p < log_p_tolerance_value) {
                break;
            }
            // 3. gamma, gamma_sum
            gpgpu.compute(gamma_param);
            gpgpu.compute(gamma_sum_param);
            // 4. mu, pi, sigma
            gpgpu.compute(mu_pi_sigma_param);
        }
        gpgpu.clear(norm_x_param.id);
        gpgpu.clear(norm_x_sum_param.id);
        return new GMM(dist_n, mu, pi, sigma);
    }
    static EvalXMuStd(dist_n, data_n, x) {
        const gpgpu = GPGPU.CreateGPGPU();
        const dist_n_zero = new Float32Array(dist_n);
        const x_mu = new Float32Array(dist_n);
        const x_std = new Float32Array(dist_n);
        const x_mu_std_shader = `
            uniform vec3 x[${data_n}];

            in vec3 zero;
            out vec3 x_mu;
            out vec3 x_std;

            void main() {
                x_mu = zero;

                for (int k = 0; k < ${data_n}; k++) {
                    x_mu += x[k]
                }

                x_mu /= ${data_n};

                x_std = zero;

                for (int k = 0; k < ${data_n}; k++) {
                    vec3 nx = (x[k] - mu);
                    x_std += nx * nx;
                }

                x_std /= ${data_n}
                x_std = sqrt(x_std);
            }
        `;
        const x_mu_std_param = {
            id: 'x_mu_std_shader',
            vertexShader: x_mu_std_shader,
            args: {
                'x': x,
                'zero': dist_n_zero,
                'x_mu': x_mu,
                'x_std': x_std
            }
        };
        gpgpu.compute(x_mu_std_param);
        gpgpu.clear(x_mu_std_param.id);
        return [x_mu, x_std];
    }
    // 変数を初期化する
    static InitMu(dist_n, x_mu, x_std) {
        const mu_x_cluster = (0, KmeansInc_1.kmeansInc)(dist_n, data_n, x);
        const mu = mu_x_cluster[0];
        /*
        // あんまりよくない初期値
        for (let i = 0; i< dist_n; i++) {
            let norm_value = this.Rnorm();
            mu[0 + i * 3] = x_std[0] * norm_value + x_mu[0];
            mu[1 + i * 3] = x_std[1] * norm_value + x_mu[1];
            mu[2 + i * 3] = x_std[1] * norm_value + x_mu[2];
        }
        */
        return mu;
    }
    static InitPi(dist_n) {
        return new Float32Array(dist_n).fill(1.0 / dist_n);
    }
    static InitSigma(dist_n, x_std) {
        const sigma = new Array(9 * dist_n).fill(0.0);
        for (let i = 0; i < dist_n; i++) {
            sigma[0 + i * 9] = x_std[0];
            sigma[4 + i * 9] = x_std[1];
            sigma[8 + i * 9] = x_std[2];
        }
        return new Float32Array(sigma);
    }
    // 標準正規分布の乱数(Box-Muller法)
    static Rnorm() {
        return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    }
}
exports.GMM = GMM;
