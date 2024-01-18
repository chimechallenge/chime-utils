## CHiME Utils
A package for 



#### Installation

`pip install .`

#### Contribute

I recommend making a fresh conda env, you can install with <br>

`pip install -e .[dev]` 
`pip install pre-commit`
`pre-commit install --install-hooks`

### CLI 

new terminal command: 
`chime-utils`

#### Usage
1. data preparation for DiPCo:
  - When you already have DiPCo

`chime-utils dgen dipco [current_DiPCo_dir] [chime8_root_dir]/chime8/dipco`

  - When you want to download it:

`chime-utils dgen dipco [dir_to_download_DiPCo] [chime8_root_dir]/chime8/dipco --download`
