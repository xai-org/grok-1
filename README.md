# Grok-1

This repository contains JAX example code for loading and running the Grok-1 open-weights model.

Make sure to download the checkpoint and place `ckpt-0` directory in `checkpoint`.
Then, run

```shell
pip install -r requirements.txt
python run.py
```

to test the code.

The script loads the checkpoint and samples from the model on a test input.

Due to the large size of the model (314B parameters), a machine with enough GPU memory is required to test the model with the example code.
The implementation of the MoE layer in this repository is not efficient. The implementation was chosen to avoid the need for custom kernels to validate the correctness of the model.

# Downloading the weights

You can download the weights using a torrent client and this magnet link:
```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

# Contributors

Thanks goes to these wonderful contributors ([emoji key](https://allcontributors.org/docs/en/emoji-key) following [all-contributors](https://github.com/all-contributors/all-contributors) specification):

<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/ibab"><img src="https://avatars.githubusercontent.com/ibab?v=4&s=100" width="100px;" alt="ibab"/><br /><sub><b>ibab</b></sub></a><br />
      <a href="https://github.com/xai-org/grok-1/commits?author=ibab" title="Code">💻</a>
      <a href="https://github.com/xai-org/grok-1/commits?author=ibab" title="Documentation">📖</a>
      <a href="#infra-ibab" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a>
      <a href="#maintenance-ibab" title="Maintenance">🚧</a>
      <a href="#ideas-ibab" title="Ideas & Planning">🤔</a>
      <a href="#review-ibab" title="Reviewed Pull Requests">👀</a>
      <a href="#tool-ibab" title="Tools">🔧</a>
      <a href="#test-ibab" title="Tests">⚠️</a>
      <a href="#research-ibab" title="Research">🔬</a>
      </td>
      <td align="center"><a href="https://github.com/TobyPDE"><img src="https://avatars.githubusercontent.com/TobyPDE?v=4&s=100" width="100px;" alt="TobyPDE"/><br /><sub><b>TobyPDE</b></sub></a><br />
      <a href="https://github.com/xai-org/grok-1/commits?author=TobyPDE" title="Code">💻</a>
      <a href="#review-TobyPDE" title="Reviewed Pull Requests">👀</a>
      <a href="#bug-TobyPDE" title="Bug reports">🐛</a>
      <a href="#ideas-TobyPDE" title="Ideas & Planning">🤔</a>
      <a href="#maintenance-TobyPDE" title="Maintenance">🚧</a>
      <a href="#test-TobyPDE" title="Tests">⚠️</a>
      <a href="#doc-TobyPDE" title="Documentation">📖</a>
      <a href="#security-TobyPDE" title="Security">🛡️</a>
      </td>
    </tr>
  </tbody>
</table>

Contributions from those not on the research team are welcome.

# License

The code and associated Grok-1 weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of Grok-1.
