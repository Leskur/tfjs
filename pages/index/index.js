const tf = require('@tensorflow/tfjs-core')
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
      if (count === 10) {
        if (this.net) {
          // console.log(frame.data)
          this.drawPose(frame)
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
      data: new Uint8Array(frame.data),
      width: frame.width,
      height: frame.height
    }
    // console.log(imgData)
    const imgSlice = tf.tidy(() => {
      const imgTensor = tf.browser.fromPixels(imgData, 4)
      return imgTensor.slice([0, 0, 0], [-1, -1, 3])
    })
    const pose = await net.estimateSinglePose(imgSlice, {
      flipHorizontal: false
    })
    imgSlice.dispose()
    return pose
  },
  async drawPose(frame) {
    const pose = await this.detectPose(frame, this.net)
    if (pose == null || this.canvas == null) return
    // console.log(pose)
    if (pose.score >= 0.3) {
      for (i in pose.keypoints) {
        console.log(pose.keypoints[i])
      }
    }
  }
})