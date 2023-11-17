import matplotlib.pyplot as plt
import io


def image_buffer_matplotlib(fig, img_type='png'):
    """
    Converts a matplotlib figure to a buffer object, and returns it
    :param fig: pyplot figure
    :param img_type: string
    :return: io_buffer
    """
    img_buf = io.BytesIO()
    # The tight layout option is used to remove the extra white space around the figure
    # It also ensures the titles and labels are not cut off from the image
    plt.savefig(img_buf, format=img_type, bbox_inches='tight')
    return img_buf


def image_buffer_plotly(fig, img_type='png'):
    """
    Converts a plotly figure to a buffer object, and returns it
    :param fig: plotly figure
    :param img_type: string
    :return: io_buffer
    """
    img_buf = io.BytesIO()
    fig.write_image(img_buf, format=img_type)
    return img_buf
