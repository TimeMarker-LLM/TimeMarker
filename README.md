# TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability

## Introduction

Recent advancements in the realm of video-language models have predominantly focused on visual perception and reasoning, leading to less emphasis on temporal localization and detection capabilities. Current models, while trained extensively on video captioning and QA datasets, struggle with placing precise temporal references within video content. Although many Video-LLMs incorporate temporal embedding into video features, this approach still has significant drawbacks. Specifically, these models can only perceive relative time—such as the sequence of events rather than absolute time points, like the exact second an event occurs. This lack of precise temporal grounding leads to less interpretable and verifiable responses, and poses challenges for subsequent temporal reasoning and inference. To address these limitations, we present **TimeMarker**, a versatile Video-LLM designed for high-quality dialogue based on video content, featuring robust temporal localization abilities.

**Key Innovations**:

1. **Temporal Separator Tokens Integration**: TimeMarker employs a method called *Temporal Separator Tokens Integration* to enhance temporal awareness in videos. By interleaving textual temporal separator tokens (e.g., sec{20}) with video frame tokens, this method explicitly encodes the absolute temporal positions of video frames. The temporal separator tokens serve as precise markers of time, allowing the model to clearly identify and reference specific moments within the video. Sequentially encoding video frames and inserting these temporal separator tokens provides a foundational basis for the LLM to accurately perceive and localize temporal information, significantly enhancing its temporal localization capabilities.

2. **AnyLength Mechanism**: To efficiently process videos of varying lengths (ranging from a few seconds to hours), TimeMarker employs the AnyLength mechanism, which utilizes *dynamic frame sampling* and *adaptive token resizing/merging*. Based on the LLM's context length, this mechanism samples frames from the video and limits the maximum number of frames input to the LLM. For short videos, we increase the frames per second (FPS) and reduce the compression ratio of single-frame tokens in our adaptive token merge module. Conversely, for long videos, we decrease the FPS and increase the compression ratio of single-frame tokens to ensure comprehensive coverage across different video durations. Meanwhile, the use of temporal separators ensures that even with dynamic changes in the number of tokens per frame, the LLM can accurately interpret the temporal information without confusion.

3. **Advanced Data Utilization**: Beyond conventional video captioning and QA datasets, we convert annotations from temporal action detection, action segmentation, video summarization, and temporal sentence grounding datasets into temporal-related video QA datasets. Temporal expressions are uniformly adapted to our tokenized format (such as temporal separator tokens), facilitating comprehensive model training on temporal understanding tasks. Despite using a relatively modest amount of video data compared to other Video-LLMs, with approximately *5M* video-text pairs, our training video is highly diverse in duration, ranging from short clips of less than one minute to lengthy videos up to 120 minutes. Additionally, we leverage extensive image training data, totaling around *90M* image data, and *12M* interleaved multi-image data. This diverse and extensive dataset enhances the model's foundational semantic perception and cognitive abilities, as well as its understanding of complex scenes.
   
4. **Benchmark Excellence Across Various Video Lengths**: Our evaluations demonstrate that TimeMarker achieves state-of-the-art performance across multiple public video benchmarks, excelling in both short and long video categories. Notably, TimeMarker surpasses traditional specialized models in tasks such as temporal sentence grounding, highlighting its superior temporal localization and understanding capabilities. This underscores the model's robustness and versatility in handling videos of varying lengths with exceptional accuracy in time-based tasks.


## News
[2024/10/30] 🔥 We release our TimeMarker 1.0 model, and paper is coming soon.


## Model Architecture
<img width="1260" alt="截屏2024-10-29 14 35 48" src="https://github.com/user-attachments/assets/046eac5c-03d4-4f61-a95c-859f2c984177">



## Performance
### Results on Video Benchmarks
| Model Name               | LLM              | MVBench | TempCompass | VideoMME (w/o subs) | VideoVista | MLVU | LVbench | LongVideoBench |
|--------------------------|------------------|---------|-------------|---------------------|------------|------|---------|----------------|
| Gemini-1.5-pro           | -                | -       | 67.1        | 75.0                | 76.4       | -    | 33.1    | 66.4           |
| GPT-4V                   | -                | 43.7    | -           | 59.9                | -          | 49.2 | -       | 60.7           |
| GPT-4o                   | -                | -       | -           | 71.9                | 78.3       | 64.6 | 27.0    | 66.7           |
| [LLaVA-Next-Video-7B](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B)      | Vicuna-7b-v1.5   | 53.1    | -           | 33.7                | 56.7       | -    | -       | 43.5           |
| [PLLaVA-7B](https://github.com/magic-research/PLLaVA)                | Vicuna-7b-v1.5   | 46.6    | -           | -                   | 60.4       | -    | -       | 39.2           |
| [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA)                   | Qwen2-7B         | -       | -           | 52.6                | 67.4       | 56.3 | -       | -              |
| [Qwen2-VL-7B-Instruct](https://github.com/QwenLM/Qwen2-VL)     | Qwen2-7B         | 67.0    | 67.8        | 63.3                | -          | -    | -       | -              |
| [VideoChat2-HD](https://github.com/OpenGVLab/Ask-Anything)          | Mistral-7B       | 62.3    | -           | 54.8                | -          | 47.9 | -       | -              |
| [VideoLLaMA2-7B](https://github.com/DAMO-NLP-SG/VideoLLaMA2)           | Mistral-7B       | 54.6    | -           | 47.9                | 60.5       | 48.5 | -       | -              |
| [Kangaroo](https://github.com/KangarooGroup/Kangaroo)                 | Llama3-8B        | 61.1    | 62.5        | 56.0                | 69.5       | 61.0 | 39.4    | 54.8           |
| TimeMarker(Ours)       | Llama3-8B        | 67.4    | 61.2        | 59.3                | 78.4       | 63.9 | 41.3    | 56.3           |




### Results on Temporal Sentence Grounding Benchmarks
<table>
   <tr>
      <th rowspan="2" style="width: 100px;">Model Name</th>
      <th rowspan="2" style="width: 100px;">Set up</th>
      <th colspan="4" style="text-align: center; width: 400px;">Charades-STA</th>
      <th colspan="4" style="text-align: center; width: 400px;">ActivityNetCaptions</th>
      <th colspan="4" style="text-align: center; width: 400px;">Didemo</th>
   </tr>
   <tr>
      <th style="width: 100px;">R@1,IoU=0.3</th>
      <th style="width: 100px;">R@1,IoU=0.5</th>
      <th style="width: 100px;">R@1,IoU=0.7</th>
      <th style="width: 100px;">mIoU</th>
      <th style="width: 100px;">R@1,IoU=0.3</th>
      <th style="width: 100px;">R@1,IoU=0.5</th>
      <th style="width: 100px;">R@1,IoU=0.7</th>
      <th style="width: 100px;">mIoU</th>
      <th style="width: 100px;">R@1,IoU=0.3</th>
      <th style="width: 100px;">R@1,IoU=0.5</th>
      <th style="width: 100px;">R@1,IoU=0.7</th>
      <th style="width: 100px;">mIoU</th>
   </tr>
   <tr>
      <td style="width: 100px;"><a href="https://github.com/researchmm/2D-TAN">2D-TAN</a></td>
      <td style="width: 100px;">FS</td>
      <td style="width: 100px;">57.3</td><td style="width: 100px;">45.8</td><td style="width: 100px;">27.9</td><td style="width: 100px;">41.0</td>
      <td style="width: 100px;">60.4</td><td style="width: 100px;">43.4</td><td style="width: 100px;">25.0</td><td style="width: 100px;">42.5</td>
      <td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td>
   </tr>
   <tr>
      <td style="width: 100px;"><a href="https://github.com/MCG-NJU/MMN">MMN</a></td>
      <td style="width: 100px;">FS</td>
      <td style="width: 100px;">65.4</td><td style="width: 100px;">53.3</td><td style="width: 100px;">31.5</td><td style="width: 100px;">46.5</td>
      <td style="width: 100px;">64.5</td><td style="width: 100px;">48.2</td><td style="width: 100px;">29.4</td><td style="width: 100px;">46.6</td>
      <td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td>
   </tr>
   <tr>
      <td style="width: 100px;"><a href="https://github.com/showlab/UniVTG">UniVTG</a></td>
      <td style="width: 100px;">FS</td>
      <td style="width: 100px;">72.6</td><td style="width: 100px;">60.2</td><td style="width: 100px;">38.6</td><td style="width: 100px;">52.2</td>
      <td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td>
      <td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td>
   </tr>
   <tr>
      <td style="width: 100px;"><a href="https://github.com/DCDmllm/Momentor">Momentor</a></td>
      <td style="width: 100px;">VLM</td>
      <td style="width: 100px;">42.6</td><td style="width: 100px;">26.6</td><td style="width: 100px;">11.6</td><td style="width: 100px;">28.5</td>
      <td style="width: 100px;">42.9</td><td style="width: 100px;">23.0</td><td style="width: 100px;">12.4</td><td style="width: 100px;">29.3</td>
      <td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td>
   </tr>
   <tr>
      <td style="width: 100px;"><a href="https://openaccess.thecvf.com/content/CVPR2024W/PVUW/papers/Qu_ChatVTG_Video_Temporal_Grounding_via_Chat_with_Video_Dialogue_Large_CVPRW_2024_paper.pdf">ChatVTG</a></td>
      <td style="width: 100px;">VLM</td>
      <td style="width: 100px;">52.7</td><td style="width: 100px;">33.0</td><td style="width: 100px;">15.9</td><td style="width: 100px;">34.9</td>
      <td style="width: 100px;">40.7</td><td style="width: 100px;">22.5</td><td style="width: 100px;">9.4</td><td style="width: 100px;">27.2</td>
      <td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td>
   </tr>
   <tr>
      <td style="width: 100px;"><a href="https://github.com/huangb23/VTimeLLM">VTimeLLM</a></td>
      <td style="width: 100px;">VLM</td>
      <td style="width: 100px;">55.3</td><td style="width: 100px;">34.3</td><td style="width: 100px;">14.7</td><td style="width: 100px;">34.6</td>
      <td style="width: 100px;">44.8</td><td style="width: 100px;">29.5</td><td style="width: 100px;">14.2</td><td style="width: 100px;">31.4</td>
      <td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td><td style="width: 100px;">-</td>
   </tr>
   <tr>
      <td style="width: 100px;">TimeMarker(Ours)</td>
      <td style="width: 100px;">VLM</td>
      <td style="width: 100px;">73.5</td><td style="width: 100px;">51.9</td><td style="width: 100px;">26.9</td><td style="width: 100px;">48.4</td>
      <td style="width: 100px;"></td><td style="width: 100px;"></td><td style="width: 100px;"></td><td style="width: 100px;"></td>
      <td style="width: 100px;">71.3</td><td style="width: 100px;">63.9</td><td style="width: 100px;">56.2</td><td style="width: 100px;">63.6</td>
   </tr>
</table>
Note: FS means the model is a specialized model for temporal sentence grounding in video trained in a fully supervised setting, VLM means the model is a Video-LLM.



