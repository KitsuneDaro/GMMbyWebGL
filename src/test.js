const gpgpu = CreateGPGPU();

const A = new Float32Array([1, 1, 1, 1]);
const B = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]);

const shader = `
    in float A;
    uniform sampler2D B;
    out float O;

    void main() {
        int x = gl_VertexID % 2;
        int y = gl_VertexID / 2;
        float b = float(texelFetch(B, ivec2(x, y), 0).r);

        O = A * b;
    }
`;
const param = {
    id: 'test',
    vertexShader: shader,
    args: {
        'A': A,
        'B': gpgpu.makeTextureInfo('float', [3, 3], B),
        'O': A
    }
}

gpgpu.compute(param);
console.log(A);

gpgpu.compute(param);
console.log(A);

gpgpu.clear(param.id);