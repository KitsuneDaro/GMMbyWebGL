"use strict";
const gpgpu = CreateGPGPU();
//const A = new Float32Array(6);
const B = new Float32Array([
    1.0, 2.0, 3.0,
]);
const D = new Float32Array([
    1.0, 2.0, 3.0,
    1.1, 2.2, 3.3,
    1.3, 2.2, 3.1,
    4.0, 2.0, 2.0,
    4.3, 2.2, 2.1,
    4.1, 2.2, 2.2
]);
const O = new Float32Array(3).fill(1.2);
const C = new Float32Array(3);
const shader = `
    in float B;
    out float C[3];

    void main() {
        C[0] = B;
        C[1] = B + 1.0;
        C[2] = B + 2.0;
    }
`;
const param = {
    id: 'test',
    vertexShader: shader,
    args: {
        'B': B,
        'C': C
    }
};
gpgpu.compute(param);
console.log(C);
gpgpu.compute(param);
console.log(C);
gpgpu.clear(param.id);
