import typing as tp
from io import BytesIO


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests

# headers are needed to get 200 from request
headers = {
    "User-Agent": """\
        Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
        AppleWebKit/537.36 (KHTML, like Gecko) \
        Chrome/85.0.4183.121 Safari/537.36\
    """
}


def chunks(array, n):
    """Return generator that yields chunks."""
    for i in range(0, len(array), n):
        yield array[i:i + n]


def rec_imaging(
    product_ids: tp.List[int],
    content_dict: tp.Dict[int, tp.Dict[str, int]],
    measure: tp.List[float] = None,
    top_n: int = 5
) -> None:
    """Render multiple images."""
    picture_urls = [content_dict[i]["image_url"] for i in product_ids]
    cnt = 0
    for _, chunk in enumerate(chunks(picture_urls, top_n)):
        fig = plt.figure(figsize=(20, 4))
        for n, i in enumerate(chunk):
            r = requests.get(i, headers=headers)
            im = Image.open(BytesIO(r.content))

            a = fig.add_subplot(1, top_n, n + 1)
            if measure is not None:
                a.title.set_text("measure = {}\n{}...".format(
                    np.round(measure[cnt], 4),
                    content_dict[product_ids[cnt]]["name"][:30]
                ))
                cnt += 1
            else:
                a.title.set_text(content_dict[product_ids[cnt]]["name"])
                cnt += 1
            plt.imshow(im)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
