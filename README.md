# synthetic_attention

## Usage
```python
batch_size, max_length, max_length = 10, 100, 128

layer = SyntheticAttention(d_model, max_length)

x = torch.randn(batch_size, max_length - 10, max_length)

output = layer(x)
```

## Reference
```bibtex
@misc{tay2021synthesizer,
      title={Synthesizer: Rethinking Self-Attention in Transformer Models}, 
      author={Yi Tay and Dara Bahri and Donald Metzler and Da-Cheng Juan and Zhe Zhao and Che Zheng},
      year={2021},
      eprint={2005.00743},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
