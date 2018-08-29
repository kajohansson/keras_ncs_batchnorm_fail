# keras_ncs_batchnorm_fail

Simple dummy network to showcase building a keras network that I can't get to run with NCS.

![example network graph](https://raw.githubusercontent.com/kajohansson/keras_ncs_batchnorm_fail/master/plotmodel_example.png "example network graph")

To create, train and export model using keras, run:

```./run.sh```

If I then run the mvNCCompile on the produced meta file it produces the following error message:

```$ mvNCCompile example.chkp.meta -in=input_1 -on=conv5/Relu -is 160 160
mvNCCompile v02.00, Copyright @ Intel Corporation 2017

/usr/local/lib/python3.5/dist-packages/tensorflow/python/util/tf_inspect.py:45: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead
shape: [1, 160, 160, 3]
[Error 5] Toolkit Error: Stage Details Not Supported: IsVariableInitialized
```

If I run it on the frozen and optimized for inference protobuf, it produces this:

```$ mvNCCompile opt_example.pb -in=input_1 -on=conv5/Relu -is 160 160
mvNCCompile v02.00, Copyright @ Intel Corporation 2017

/usr/local/lib/python3.5/dist-packages/tensorflow/python/util/tf_inspect.py:45: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead
shape: [1, 160, 160, 3]
[Error 5] Toolkit Error: Stage Details Not Supported: Top Not Supported - Constants conv1_bn/moving_variance
```

Also worth noting, if I run the network with another input size if produces this for the protobuf:

```$ mvNCCompile opt_example.pb -in=input_1 -on=conv5/Relu -is 240 240
mvNCCompile v02.00, Copyright @ Intel Corporation 2017

/usr/local/lib/python3.5/dist-packages/tensorflow/python/util/tf_inspect.py:45: DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead
shape: [1, 240, 240, 3]
Traceback (most recent call last):
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py", line 1327, in _do_call
    return fn(*args)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py", line 1312, in _run_fn
    options, feed_dict, fetch_list, target_list, run_metadata)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py", line 1420, in _call_tf_sessionrun
    status, run_metadata)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/errors_impl.py", line 516, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [1,120,120,64] vs. [1,80,1,1]
	 [[Node: conv1_bn/batchnorm_1/mul_1 = Mul[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"](conv1/Relu, conv1_bn/batchnorm_1/mul)]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/bin/mvNCCompile", line 169, in <module>
    create_graph(args.network, args.image, args.inputnode, args.outputnode, args.outfile, args.nshaves, args.inputsize, args.weights, args.explicit_concat, args.ma2480, args.scheduler, args.new_parser, args)
  File "/usr/local/bin/mvNCCompile", line 148, in create_graph
    load_ret = load_network(args, parser, myriad_config)
  File "/usr/local/bin/ncsdk/Controllers/Scheduler.py", line 100, in load_network
    parse_ret = parse_tensor(arguments, myriad_conf)
  File "/usr/local/bin/ncsdk/Controllers/TensorFlowParser.py", line 294, in parse_tensor
    res = outputTensor.eval(feed_dict)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/ops.py", line 656, in eval
    return _eval_using_default_session(self, feed_dict, self.graph, session)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/ops.py", line 5016, in _eval_using_default_session
    return session.run(tensors, feed_dict)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py", line 905, in run
    run_metadata_ptr)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py", line 1140, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py", line 1321, in _do_run
    run_metadata)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py", line 1340, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [1,120,120,64] vs. [1,80,1,1]
	 [[Node: conv1_bn/batchnorm_1/mul_1 = Mul[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"](conv1/Relu, conv1_bn/batchnorm_1/mul)]]

Caused by op 'conv1_bn/batchnorm_1/mul_1', defined at:
  File "/usr/local/bin/mvNCCompile", line 169, in <module>
    create_graph(args.network, args.image, args.inputnode, args.outputnode, args.outfile, args.nshaves, args.inputsize, args.weights, args.explicit_concat, args.ma2480, args.scheduler, args.new_parser, args)
  File "/usr/local/bin/mvNCCompile", line 148, in create_graph
    load_ret = load_network(args, parser, myriad_config)
  File "/usr/local/bin/ncsdk/Controllers/Scheduler.py", line 100, in load_network
    parse_ret = parse_tensor(arguments, myriad_conf)
  File "/usr/local/bin/ncsdk/Controllers/TensorFlowParser.py", line 212, in parse_tensor
    tf.import_graph_def(graph_def, name="")
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/util/deprecation.py", line 432, in new_func
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/importer.py", line 577, in import_graph_def
    op_def=op_def)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/ops.py", line 3290, in create_op
    op_def=op_def)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/ops.py", line 1654, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): Incompatible shapes: [1,120,120,64] vs. [1,80,1,1]
	 [[Node: conv1_bn/batchnorm_1/mul_1 = Mul[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"](conv1/Relu, conv1_bn/batchnorm_1/mul)]]
```

Any help with how to get BatchNormalization to work with Keras and Movidius NCS is highly appreciated!

