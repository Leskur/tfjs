const posenet = require('@tensorflow-models/posenet')
//index.js

Page({
  async onReady() {
    const camera = wx.createCameraContext(this)
    this.canvas = wx.createCanvasContext('pose', this)
    this.net = await posenet.load({

    })

    let count = 0
    const listener = camera.onCameraFrame((frame) => {
      count++
      if (count === 4) {
        console.log(frame.data)
        count = 0
      }

    })
    listener.start()

  },
})