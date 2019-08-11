const tf = require('@/tensorflow/tfjs-core')
const posenet = require('@tensorflow-models/posenet')


//index.js

Page({
  async onReady() {
    const camera = wx.createCameraContext(this)
    this.canvas = wx.createCanvasContext('pose', this)
    this.loadPosenet()
    let count = 0
    const listener = camera.onCameraFrame((frame) => {
      count++
      if (count === 4) {
        if (this.net) {
          this.detectPose(frame, this.net)
          console.log(frame.data)
        }
        count = 0
      }

    })
    // listener.start()
  },
  async loadPosenet() {
    this.net = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: 193,
      multiplier: 0.5,
      // modelUrl: 'https://www.gstaticcnapps.cn/tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json'
    })
    console.log(this.net)
  },
  async detectPose(frame, net) {
    const imgData = {
      data: new Uint32Array(frame.data),
      width: frame.width,
      height: frame.height
    }
    const imgTensor = tf.browser.fromPixels(imgData, 4)
    const imgSlice = imgTensor.slice([0, 0, 0], [-1, -1, 3])
  }
})