import tensorflow as tf

class WriteSummary():
    def __init__(self, writePath):
        self.writer = tf.summary.FileWriter(writePath)
        tf.summary.scalar("Loss", loss_)
        write_op = tf.summary.merge_all()

    def __call__(self):
        summaryWrite = self.writer.flush()
        return summaryWrite

class SaveModel():
    def __init__(self, savePath):
        self.saver = tf.train.Saver()
        self.savePath = savePath
    def __call__(self, model):
        modelSave = self.saver.save(model, self.savePath)
        return modelSave
        
