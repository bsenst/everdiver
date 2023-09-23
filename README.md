# Enhancing Your Personal Notes with LLMs - LabLab.ai Falcon LLM Hackathon

falcon offers the chance to deploy an open-source large language model locally. this allows to preserve privacy working with your personal evernote data. as such this project is part of the [lablabai falcon hackathon](https://lablab.ai/event/falcon-llms-24-hours-hackathon/fritzlabs).

- [X] create dummy evernote data
- [X] extract data from evernote exported enex file
- [X] deploy/serve falcon llm locally or in private free cloud environment (kaggle, colab, saturncloud) or connect to hosting service (clarifai)
- [X] connect enex data to deployed llm using tools such as llamaindex or langchain

![image](https://github.com/bsenst/everdiver/assets/8211411/4738a376-16fb-43a6-90b8-5f487db92e23)

```
streamlit run streamlit-app/app.py
```

# Examples of the Application

**When will be lunch break according to the schedule on Friday?**

> Lunch break will be at 12:30 pm on Friday according to the schedule.

**Which books on my reading list are authored by Jack London?**

> "The Call of the Wild", "White Fang", and "The Sea-Wolf" are all authored by Jack London.

![image](https://github.com/bsenst/everdiver/assets/8211411/30904bf9-b95e-4150-b8fe-284c07841aab)

![image](https://github.com/bsenst/everdiver/assets/8211411/df91c710-8c20-410f-a397-886d780c0124)

## References

* https://github.com/putuwaw/docutalk *Apache 2.0 open source license*
* https://www.kaggle.com/code/hinepo/q-a-chatbot-with-llms-harry-potter *Apache 2.0 open source license*

## Serving open-source LLMs

* https://github.com/bentoml/OpenLLM
* https://github.com/vllm-project/vllm
* https://github.com/skypilot-org/skypilot
* https://github.com/TimDettmers/bitsandbytes Quantization GPU

# Everdiver

Evernote does not offer the possibility to analyze your notes collection. Python can be used to access the local evernote database or exported note files and analyze the information it contains.

## Get more from your Personal Notes
- [ ] show areas of expertise
- [ ] display the depths of knowledge in each corresponding field (beginner vs expert)
- [ ] show time course of creating/updating notes in each knowledge field
- [ ] visualize connection between notes
- [ ] identify missing knowledge comparing to outside databases
