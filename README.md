# quant-kit
Quantitative Investment and Trading Toolkit


## Environment setup & installation of modules
1. Create a virtual conda environment:
    ```bash
    # Update conda
    $ conda update conda

    # Create environment
    $ conda create -n quant_kit_env -c conda-forge \
        black \
        ipykernel \
        python=3.10 \
        && conda clean -tipy
    ```

2. Clone `quant-kit` repository in your desired path:
    ```bash
    $ git clone git@github.com:libertininick/quant-kit.git
    ```

    (Optional) Update your remote origin head branch to `main`
    ```bash
    $ git symbolic-ref refs/remotes/origin/HEAD refs/remotes/origin/main
    ```

3. Install the desired modules:
    ```bash
    $ cd quant-kit
    $ conda activate quant_kit_env
    $ python -m pip install <module name>
    ```
    - [quant-kit-app](quant-kit-app/README.md)
    - [quant-kit-core](quant-kit-core/README.md)