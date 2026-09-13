# Released checkpoints

Keep only the four paper checkpoints here. Model files are intentionally not
tracked by Git; their Google Drive links are maintained in the root README.

```text
checkpoints/
├── esconv/
│   ├── hard/best.pth
│   └── soft/best.pth
└── annomi/
    ├── hard/best.pth
    └── soft/best.pth
```

`hard` maps to `original_hard` in the implementation. `soft` maps to
`learnable_soft`. Copy each validation-best file to `best.pth` before release;
do not expose training-step filenames as the public interface.
