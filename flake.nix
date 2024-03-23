{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python311;
    in
    {
      devShells.default = pkgs.mkShell {

        # build-time
        nativeBuildInputs = with pkgs; [
          bashInteractive
        ];

        # run-time
        buildInputs = (with pkgs; [
          transmission
          just
        ]) ++ (with python.pkgs; [
          setuptools
          wheel
          venvShellHook
          pylint
        ]);

        # python setup
        src = null;
        venvDir = ".venv";
        postVenv = ''
          unset SOURCE_DATE_EPOCH
        '';
        postShellHook = ''
          # python setup
          unset SOURCE_DATE_EPOCH
          unset LD_PRELOAD
          PYTHONPATH=$PWD/$venvDir/${python.sitePackages}:$PYTHONPATH
          pip install --require-virtualenv -r requirements.txt | grep -v 'already satisfied'
        '';

      };
    }
  );
}
