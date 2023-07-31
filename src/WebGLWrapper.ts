export class WebGLWrapper {
    static createShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
        // シェーダーを作成
        var shader = <WebGLShader>gl.createShader(type);

        // GLSLのコードをGPUにアップロード
        gl.shaderSource(shader, source);
        // シェーダーをコンパイル
        gl.compileShader(shader);
        // 成功かどうかチェック

        var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
        if (success) {
            return shader; // 成功。シェーダーを返す
        }

        // シェーダーを削除
        gl.deleteShader(shader);

        // エラーを表示
        throw new Error(<string>gl.getShaderInfoLog(shader));
    }

    static createTexture2D(gl: WebGLRenderingContext, data: number[], width: number, height: number): WebGLTexture {
        const ext = <WEBGL_color_buffer_float>gl.getExtension("WEBGL_color_buffer_float");

        const float_data = new Float32Array(data);
        const texture = <WebGLTexture>gl.createTexture();

        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, ext.RGBA32F_EXT, width, height, 0, gl.ALPHA, gl.FLOAT, float_data);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)

        gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

        // 縮小拡大時のテクスチャの利用値をぼかしなしにする
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        // 縦横にはみ出したら0.0か1.0とする
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        return texture;
    }

    static createGL(canvas: HTMLCanvasElement): WebGLRenderingContext {
        const gl = <WebGLRenderingContext>canvas.getContext("webgl");

        return gl;
    }
}
