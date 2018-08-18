import matplotlib.pyplot as plt



def plot_acc_loss(history, metrics):
    plt.subplot(121)
    plt.plot(history.history[metrics])
    plt.plot(history.history['val_'+metrics])
    plt.title('model accuracy')
    plt.ylabel(metrics)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.subplots_adjust()
    plt.show()


def plot_acc_loss_all(history_array, metrics):
    for hist in history_array:
        plt.subplot(121)
        plt.plot(hist.history[metrics])
        plt.plot(hist.history['val_' + metrics])
        plt.title('model accuracy')
        plt.ylabel(metrics)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(122)
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.subplots_adjust()
    plt.show()