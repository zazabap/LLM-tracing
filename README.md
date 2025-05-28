# LLM-tracing

I would like to design an experiment to explore the content/security of certain layers during the fine-tuning. Based on following principles: 

1. The fine-tuning dataset (related with Robustness): TruthfulQA, ToxiGen, SST-2, ConfAIde.
2. The infinit-gram. is based on llama-2, I think it's better to use llama-2 model 
3. Fine-tuning method: Full Supervised Fine Tuning, DPO Alignment. Parameter Efficient Fine-Tuning. 
4. Layer options: Layer-0,6,12,18,24,30. Last Layer is 32 
5. What to calculate? 
	1. N-gram for fine-tuning dataset and the output. What is included in the layer's output from fine-tuning's input?
	2. Mutual Information. 
	3. Bound from linear probing. 

The GPT answered version does not exactly match my ideas, here are several steps for implementation. The To-Do list to record the implementation progress： 
- [x] llama3 lora fine-tune code 
	- [x] unsloth 8bit version of the model works fine. 
	- [x] some comparison functions among base model and fine-tuned model 
	- [x] Layer-wise evaluation for 0,8,16,24,31
- [ ] Using Security dataset and consider the security metrics
	- [ ] Fine-tuning with TruthfulQA, 
	- [ ] ToxiGen, 
	- [ ] SST-2, 
	- [ ] ConfAIde.
- [ ] Implement Evaluation Method: 
	- [ ] N-gram evaluation for each layer 
	- [ ] Mutual Information Computation. 