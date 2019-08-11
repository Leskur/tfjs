//index.js

Page({
  onLoad: function() {
    const camera = wx.createCameraContext(this)
    let count = 0
    const listener = camera.onCameraFrame((frame) => {
      count++
      if (count === 4) {
        console.log(frame.data)
        count = 0
      }

    })
    listener.start()
    this.canvas = wx.createCanvasContext('pose', this)
  },
})