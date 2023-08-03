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
exports.kmeans = exports.kmeansInc = void 0;
const GPGPU = __importStar(require("./gpgpu"));
function kmeansInc(dist_n, data_n, x) {
    /* k-means++ class (3D限定) */
    const gpgpu = GPGPU.CreateGPGPU();
    // Shaders
    const kmeans_inc_distance_shader = `
        uniform vec3 center;

        in vec3 x;
        out float distance2;

        void main(){
            vec3 delta = x - center;
            distance2 = dot(delta, delta);
        }
    `;
    // Variables
    const distance2 = new Float32Array(data_n);
    const first_index = Math.floor(Math.random() * dist_n);
    const indexs = new Float32Array(dist_n);
    const centers = new Float32Array(3 * dist_n);
    const center = x.slice(first_index * 3, first_index * 3 + 3);
    // 既に中心点に選んだデータ点を次の中心点に使用しないようにマスキングを行ったデータ
    const mask_x = x.slice();
    indexs[0] = first_index;
    centers.set(center, 0);
    // Parameters
    const kmeans_inc_distance_param = {
        id: 'kmeans_inc_distance_shader',
        vertexShader: kmeans_inc_distance_shader,
        args: {
            'center': center,
            'x': mask_x,
            'distance2': distance2
        }
    };
    const sum_func = (accumulator, currentValue) => {
        return accumulator + currentValue;
    };
    for (let k = 1; k < dist_n; k++) {
        // distance2を計算
        gpgpu.compute(kmeans_inc_distance_param);
        // indexを累積和から計算
        let index = data_n - 1; // 浮動小数点によってdistance2の和が1.0にならなかったときのための初期値
        let distance2_sum = distance2.reduce(sum_func);
        let distance2_cumsum = 0;
        let random = Math.random() * distance2_sum;
        for (var i = 0; i < data_n; i++) {
            distance2_cumsum += distance2[i];
            if (random < distance2_cumsum) {
                index = i;
                break;
            }
        }
        indexs[k] = index;
        // centerに求めた点を追加
        center.set(x.slice(index * 3, index * 3 + 3), 0);
        centers.set(center, k * 3);
        // 既に中心点に選んだデータ点を次の中心点に使用しないようにマスキング
        for (var i = 0; i < k; i++) {
            mask_x.set(center, indexs[i] * 3);
        }
    }
    gpgpu.clear(kmeans_inc_distance_param.id);
    return kmeans(dist_n, data_n, x, centers);
}
exports.kmeansInc = kmeansInc;
function kmeans(dist_n, data_n, x, init_centers) {
    /* k-means class (3D限定) */
    const gpgpu = GPGPU.CreateGPGPU();
    // Shaders
    const kmeans_clustering_shader = `
        uniform vec3 centers[${dist_n}];

        in vec3 x;
        out float x_cluster; // 0 ~ dist_n - 1

        void main() {
            vec3 delta = x - centers[k];
            float min_distance2 = dot(delta, delta);
            
            x_cluster = 0;

            for (int k = 1; k < ${dist_n}; k++) {
                vec3 delta = x - centers[k];
                float distance2 = dot(delta, delta);

                if (distance2 < min_distance2) {
                    min_distance2 = distance2;
                    x_cluster = k;
                }
            }
        }
    `;
    const kmeans_centers_shader = `
        uniform vec3 x[${data_n}];
        uniform float x_cluster[${data_n}];

        in vec3 zero;
        out vec3 center;

        void main() {
            int x_cluster_n = 0;
            int k = 0;

            center = vec3(0.0);
            
            for (int i = 0; i < ${data_n}; i++) {
                k = int(x_cluster[i]);

                if (k == gl_VertexID) {
                    center += x[i];
                    x_cluster_n++;
                }
            }

            center /= float(x_cluster_n);
            center += zero;
        }
    `;
    // Variables
    const centers = init_centers.slice();
    const x_cluster = new Float32Array(data_n);
    const new_centers = new Float32Array(dist_n * 3);
    const dist_n_vec3_zero = new Float32Array(dist_n * 3);
    // Parameters
    const kmeans_clustering_param = {
        id: 'kmeans_clustering_shader',
        vertexShader: kmeans_clustering_shader,
        args: {
            'centers': centers,
            'x': x,
            'x_cluster': x_cluster
        }
    };
    const kmeans_centers_param = {
        id: 'kmeans_centers_shader',
        vertexShader: kmeans_centers_shader,
        args: {
            'x': x,
            'x_cluster': x_cluster,
            'zero': dist_n_vec3_zero,
            'center': new_centers
        }
    };
    while (true) {
        gpgpu.compute(kmeans_clustering_param);
        gpgpu.compute(kmeans_centers_param);
        for (var i = 0; i < dist_n * 3; i++) {
            if (new_centers[i] != centers[i]) {
                centers.set(new_centers.slice(), 0);
                break;
            }
        }
        if (i == dist_n * 3) {
            break;
        }
    }
    gpgpu.clear(kmeans_clustering_param.id);
    gpgpu.clear(kmeans_centers_param.id);
    return [centers, x_cluster];
}
exports.kmeans = kmeans;
