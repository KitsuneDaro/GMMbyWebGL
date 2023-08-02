const gpgpu = CreateGPGPU();

//const A = new Float32Array(6);

const shader = `
    uniform sampler2D B;

    in float zero;
    out vec3 sigma;

    void main(){
        float a = float(texelFetch(B, ivec2(0, 0), 0));

        sigma = vec3(a, a, a) * (zero + 1.0);
    }
`;

const B = new Float32Array([
    1.0, 2.0, 3.0,
]);
const zero = new Float32Array(4);
const sigma = new Float32Array(4 * 3);

const param = {
    id: 'test',
    vertexShader: shader,
    args: {
        'B': gpgpu.makeTextureInfo('float', [3, 1], B),
        'zero': zero,
        'sigma': sigma
    }
};

gpgpu.compute(param);
console.log(sigma);

B[0] = 20;

gpgpu.compute(param);
console.log(sigma);

gpgpu.clear(param.id);