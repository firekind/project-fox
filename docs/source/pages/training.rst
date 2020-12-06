Training
========

.. toctree::
    :hidden:

This section covers the work done to get the model to train properly. While constructing the training code, couple of things were kept in mind:

1. To train as fast as possible, the pretrained weights from all the models have to be used. This was made possible with the way how the model is structured.
2. Reuse as much of the code from YoloV3, MiDAS and PlaneRCNN repos as possible. Don't reinvent the wheel.

MiDAS training
--------------

Training MiDAS was not much of an issue, mostly due to the fact that the MiDAS model is frozen. However, for the sake of completeness, the loss calculation for MiDAS was implemented. The loss used was a simple RMSE Loss, which was chosen by looking at the submissions of EVA4 Phase 1.

In the dataset, the input images were letterboxed, so before applying the loss function, the letterbox paddings were removed (using the padding extents that were returned by the dataset), output was resized to match the ground truth size and then the RMSE loss was applied.

.. code-block:: python

    def rmse_loss(output, target, letterbox_borders = None):
        if letterbox_borders is not None:
            output = crop_and_resize(output, letterbox_borders, target.shape[1:])
        
        return torch.sqrt(
            F.mse_loss(output, target)
        )


    def crop_and_resize(output, letterbox_borders, target_shape):
        new_output = torch.zeros(output.size(0), target_shape[0], target_shape[1], target_shape[2], device=output.device)

        for i, o in enumerate(output):
            dw, dh = letterbox_borders[i]
            if not (dw == 0 and dh == 0):
                if dw == 0:
                    dh = int(np.ceil(dh))
                    o = o[dh:-dh, :]
                else:
                    dw = int(np.ceil(dw))
                    o = o[:, dw:-dw]
            new_output[i] = F.interpolate(
                o.unsqueeze(0).unsqueeze(0), # adding channel and batch dimension
                (target_shape[1], target_shape[2])
            ).squeeze(0).squeeze(0) # removing channel and batch dimension

        return new_output

YoloV3 training
---------------

Constructing the training code for YoloV3 involved reading and understanding the gist of what the training code of vanilla YoloV3 did. refactoring the various sections of the vanilla training loop into functions that could be used easily with pytorch lightning was simple, although there were issues with when to step the optimizer and EMA which was used during validation. The loss function used for YoloV3 is the same as what is used in vanilla YoloV3. `Here <https://github.com/firekind/yolov3/tree/eva5/eva5_helper.py>`_'s the link to the refactored code that was used to train YoloV3.

PlaneRCNN training
------------------

Understanding the training loop of vanilla PlaneRCNN took sometime, but treating sections of the loop as a black box without getting into details help speed up the refactoring process. Issues encountered here were mostly related to depth and GPU vs CPU, but that was solved by disabling depth prediction in PlaneRCNN config and writing a bit of code to transfer the data to the correct device. The loss function used for PlaneRCNN is the same as what is used in vanilla PlaneRCNN. Some debate was there regarding whether the refinement part of the training was to be used, but after deliberation (and due to lack of time 😅), it was left as is. `Here <https://github.com/firekind/planercnn/tree/eva5/eva5_helper.py>`_'s the link to the refactored code that was used to train PlaneRCNN.

Putting it all together
-----------------------

The total loss was taken as a weighted sum of all the three individual losses. Now comes the first hiccup: turns out PlaneRCNN cannot be trained with a batch size of more than 1. This was found out by the presence of a lot of comments in the code of vanilla PlaneRCNN, and also quite a few calculations discard the batch dimension altogether. The second hiccup: For some reason there are hard coded slices in some calculations in PlaneRCNN, and these hard coded values could have disastrous effect if the input image size is small, since theres no telling what it would do. These two hiccups were among the primary reasons that the Yolo segment of the model was trained separately from the PlaneRCNN segment.

The first hiccup was solved by implementing something akin to gradient accumulation, by taking a batch size more than one but performing the calculations in a loop one after the other for each element in the batch (forward prop could work directly on the batch). The second hiccup couldn't be solved, and was left for future improvement.

As mentioned previously, the yolo and PlaneRCNN sections of the model were trained separately. This sped up the whole training process, and the checkpoint files could simply be merged to get the final set of pretrained weights.

The training code can be found `here <https://github.com/firekind/project-fox/blob/master/fox/model.py>`_, the notebook that was used to train the Yolo section can be found `here <https://github.com/firekind/project-fox/blob/master/yolo_train.ipynb>`_, and the notebook that was used to train the PlaneRCNN section can be found `here <https://github.com/firekind/project-fox/blob/master/planercnn_train.ipynb>`_