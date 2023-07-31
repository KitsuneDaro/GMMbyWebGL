"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const GMM_1 = require("./GMM");
window.addEventListener('onload', () => {
    const data = [
        1.0, 2.0, 3.0,
        1.1, 2.2, 3.3,
        1.3, 2.2, 3.1,
        4.0, 2.0, 2.0,
        4.3, 2.2, 2.1,
        4.1, 2.2, 2.2,
    ];
    const gmm = GMM_1.GMM.CreateGMM(2, 6, new Float32Array(data));
});
