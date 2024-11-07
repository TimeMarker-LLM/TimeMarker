# TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability

<img width="900" src="https://github.com/TimeMarker-LLM/TimeMarker/blob/main/assets/logo_w_name.jpg">

## Introduction

Recent advancements in the realm of video-language models have predominantly focused on visual perception and reasoning, leading to less emphasis on temporal localization and detection capabilities. Current models, while trained extensively on video captioning and QA datasets, struggle with placing precise temporal references within video content. Although many Video-LLMs incorporate temporal embedding into video features, this approach still has significant drawbacks. Specifically, these models can only perceive relative time—such as the sequence of events rather than absolute time points, like the exact second an event occurs. This lack of precise temporal grounding leads to less interpretable and verifiable responses, and poses challenges for subsequent temporal reasoning and inference. To address these limitations, we present **TimeMarker**, a versatile Video-LLM designed for high-quality dialogue based on video content, featuring robust temporal localization abilities.

**Key Innovations**:

1. **Temporal Separator Tokens Integration**: TimeMarker uses Temporal Separator Tokens Integration to enhance temporal awareness in videos. By interleaving textual temporal separator tokens (e.g., sec{20}) with video frame tokens, this method encodes the absolute temporal positions of video frames. These tokens serve as precise time markers, allowing the model to identify and reference specific moments within the video.

2. **AnyLength Mechanism**: To process videos of varying lengths efficiently, TimeMarker employs the AnyLength mechanism, which uses dynamic frame sampling and adaptive token resizing/merging. This mechanism adjusts the frames per second (FPS) when sample video frames and modify token compression ratio when adaptively merge tokens in a single video frame based on the video's length, ensuring comprehensive coverage of various-length videos. 

3. **Advanced Data Utilization**: Beyond conventional video captioning and QA datasets, TimeMarker converts annotations from various temporal-related datasets into video QA formats, facilitating comprehensive model training on temporal understanding tasks. Despite using only approximately 5M video-text pairs in training, the video data is diverse in duration, from less than one minute to 120 minutes. Additionally, extensive image training data (around 90M) and interleaved multi-image data (around 12M) enhance the model's semantic perception and cognitive abilities.
   
4. **Benchmark Excellence Across Various Video Lengths**: TimeMarker achieves state-of-the-art performance across multiple public video benchmarks, excelling in both short and long video categories. It surpasses traditional models in tasks such as temporal sentence grounding, demonstrating superior temporal localization and understanding capabilities. This underscores the model's robustness and versatility in handling videos of varying lengths with exceptional accuracy in time-based tasks.


## News
[2024/10/30] 🔥 We release our TimeMarker model. TimeMarker is based on Llama3-8B LLM, and achieves 🌟Rank 1 on [LVBench](https://lvbench.github.io/#leaderboard), 🌟Rank 2 on [VideoVista](https://videovista.github.io/#leaderboard) (Rank 1 on VideoVista is Human Performance), 🌟Rank 2 on [MVBench](https://huggingface.co/spaces/OpenGVLab/MVBench_Leaderboard), and 🌟Rank 3 on [MLVU test set](https://github.com/JUNJIE99/MLVU?tab=readme-ov-file#trophy-mlvu-test-leaderboard)! The results of our TimeMarker also rank highly in other video benchmarks. Our paper is coming soon.


## Model Architecture
<img width="1260" src="https://github.com/TimeMarker-LLM/TimeMarker/blob/main/assets/timemarker_framework.png">



## Performance
### Results on Video Benchmarks
| Model Name                                                                 | LLM              | [VideoMME (w/o subs)](https://video-mme.github.io/home_page.html#leaderboard) | [VideoVista](https://videovista.github.io/#leaderboard) | [LVbench](https://lvbench.github.io/#leaderboard) | [LongVideoBench (dev)](https://huggingface.co/spaces/longvideobench/LongVideoBench) | [MLVU (dev)](https://github.com/JUNJIE99/MLVU) | [MVBench](https://huggingface.co/spaces/OpenGVLab/MVBench_Leaderboard) | [MMBench-Video](https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard) | [TempCompass](https://huggingface.co/spaces/lyx97/TempCompass) |
|---------------------------------------------------------------------------|------------------|---------------------|------------|---------|----------------|------|---------|---------------|-------------|
| Gemini-1.5-pro                                                            | -                | 75.0                | 76.4       | 33.1    | 66.4           | -    | -       | 1.30          | 67.1        |
| GPT-4V                                                                    | -                | 59.9                | -          | -       | 60.7           | 49.2 | 43.7    | 1.53          | -           |
| GPT-4o                                                                    | -                | 71.9                | 78.3       | 27.0    | 66.7           | 64.6 | -       | 1.64          | -           |
| [LLaVA-Next-Video-7B](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B) | Vicuna-7b-v1.5   | 33.7                | 56.7       | -       | 43.5           | -    | 53.1    | -            | -           |
| [PLLaVA-7B](https://github.com/magic-research/PLLaVA)                     | Vicuna-7b-v1.5   | -                   | 60.4       | -       | 39.2           | -    | 46.6    | 1.03          | -           |
| [VideoChat2-HD](https://github.com/OpenGVLab/Ask-Anything)                | Mistral-7B       | -                   | 61.6       | -       | -              | 47.9 | 62.3    | 1.22          | -           |
| [VideoLLaMA2-7B](https://github.com/DAMO-NLP-SG/VideoLLaMA2)              | Mistral-7B       | 47.9                | 60.5       | -       | -              | 48.5 | 54.6    | -             | -           |
| [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA)                      | Qwen2-7B         | 52.6                | 67.4       | -       | -              | 56.3 | -       | -             | 56.9        |
| [Video-XL](https://github.com/VectorSpaceLab/Video-XL)                    | Qwen2-7B         | 55.5                | -          | -       | 49.5           | 64.9 | 55.3    | -             | -           |
| [Qwen2-VL-7B-Instruct](https://github.com/QwenLM/Qwen2-VL)                | Qwen2-7B         | 63.3                | -          | -       | -              | -    | 67.0    | -             | 67.8        |
| [Kangaroo](https://github.com/KangarooGroup/Kangaroo)                     | Llama3-8B        | 56.0                | 69.5       | 39.4    | 54.8           | 61.0 | 61.1    | 1.44          | -           |
| TimeMarker (Ours)                                                         | Llama3-8B        | 57.3                | 78.4       | 41.3    | 56.3           | 63.9 | 67.4    | 1.53          | 60.4        |



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
      <td style="width: 100px;">67.4</td><td style="width: 100px;">50.7</td><td style="width: 100px;">33.0</td><td style="width: 100px;">49.5</td>
      <td style="width: 100px;">71.3</td><td style="width: 100px;">63.9</td><td style="width: 100px;">56.2</td><td style="width: 100px;">63.6</td>
   </tr>
</table>
Note: FS means the model is a specialized model for temporal sentence grounding in video trained in a fully supervised setting, VLM means the model is a Video-LLM.



