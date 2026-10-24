"""tuningfork with no GPU and no model: `tuningfork demo`.

A real proxy over made-up recordings (`lab`), in front of a scripted model (`mock`) whose raw
endpoints are a toy llama.cpp (`toy`), with a harness it can launch (`agent`, adapted by
`task.FakeAgent`) on the demo's task (`task`). The test suite runs on the same pieces. Nothing
else in tuningfork imports them.
"""
