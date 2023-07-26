#!/bin/bash

git clone git@github.com:ijapesigan/simAutoReg.git
rm -rf "$PWD.git"
mv simAutoReg/.git "$PWD"
rm -rf simAutoReg
