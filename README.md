# PromptReel Investigator 
## Overview
This project is a derivative of the [videollama repository](https://github.com/DAMO-NLP-SG/Video-LLaMA), initiated by forking the source code and adapting the environment setup as per the provided instructions. The primary objective of this project is to compose a prompt for a specific video using a generalized few-shot prompt (prompt composition was selected from the analysis from [VideoLlama_X_OOPS_Prompt_Eng](https://github.com/Hasnat79/videollama_x_oops_prompt_eng)) appended with the timestamp of the event segment. That prompt is then input into the videollama model for producing output on a randomly selected sample of 100 videos, which are irrelevant to the prompt, from the OOPS validation set having only failed videos ([Oops_extractor](https://github.com/Hasnat79/Oops_extractor)).


## Setup

1. Environment Configuration: The environment setup for this project follows the guidelines provided in the [videollama](https://github.com/DAMO-NLP-SG/Video-LLaMA) readme file. Refer to the original repository for detailed instructions on how to configure the environment.
2. run the [__irrelevant_video_prompt_exp.py](./__irrelevant_video_prompt_exp.py)
file to execute the model on a randomly selected sample of 100 videos from the OOPS validation set, excluding the video used to compose the prompt.

## Results Data
The detailed results can be found in the [__irr_vid_prmt_exp_data.json](./results/__irr_vid_prmt_exp_data.json) file. 

## Observations and Insights
For a comprehensive understanding of the results, refer to the [PromptReel Investigator Observations](https://docs.google.com/document/d/127Dz9mZk-qVI6-2zdDWY9xFD1p-LYS-5UkN3QIGkJWw/edit?usp=sharing) document. This file contains detailed observations and insights derived from the experiment.

## Next Steps
Based on the observations, consider the following steps for further analysis or improvements to the project:
- Identify patterns or trends in the model's responses.
- Evaluate the effectiveness of the prompts in generating meaningful outputs.
- Explore potential adjustments to the experimental setup for enhanced results.



## Files and Directories
- `PromptReel_Investigator/results/__irr_vid_prmt_exp_data.json`: Results data from the model execution.
- [PromptReel Investigator Observations](https://docs.google.com/document/d/127Dz9mZk-qVI6-2zdDWY9xFD1p-LYS-5UkN3QIGkJWw/edit?usp=sharing): Detailed documentation of observations and insights.

**Note**: Ensure that the results data file is reviewed in conjunction with the observations document to derive meaningful conclusions from the experiment.
## Citation
If you use the experiment results from this repository in your work, please consider citing this project. Thank you <3

## Contact
Feel free to reach out if you have any questions or suggestions!
- 📧 Email: hasnatabdullah79@gmail.com
- 💼 LinkedIn: [Hasnat Md Abdullah ](https://www.linkedin.com/in/hasnat-md-abdullah/)