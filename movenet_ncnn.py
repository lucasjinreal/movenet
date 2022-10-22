import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(dtype=null)
    out = []

    with ncnn.Net() as net:
         net.load_param("/Users/lewisjin/work/codes/wnn/vendor/movenet/movenet.ncnn.param")
         net.load_model("/Users/lewisjin/work/codes/wnn/vendor/movenet/movenet.ncnn.bin")

         with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)
