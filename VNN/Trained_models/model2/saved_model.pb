ô
Ñµ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.22unknown8³

Adam/GO-output_mod/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameAdam/GO-output_mod/kernel/v

/Adam/GO-output_mod/kernel/v/Read/ReadVariableOpReadVariableOpAdam/GO-output_mod/kernel/v*
_output_shapes

:*
dtype0

Adam/GO-02_mod/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/GO-02_mod/bias/v
{
)Adam/GO-02_mod/bias/v/Read/ReadVariableOpReadVariableOpAdam/GO-02_mod/bias/v*
_output_shapes
:*
dtype0

Adam/GO-02_mod/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/GO-02_mod/kernel/v

+Adam/GO-02_mod/kernel/v/Read/ReadVariableOpReadVariableOpAdam/GO-02_mod/kernel/v*
_output_shapes

:*
dtype0

Adam/GO-01_mod/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/GO-01_mod/bias/v
{
)Adam/GO-01_mod/bias/v/Read/ReadVariableOpReadVariableOpAdam/GO-01_mod/bias/v*
_output_shapes
:*
dtype0

Adam/GO-01_mod/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/GO-01_mod/kernel/v

+Adam/GO-01_mod/kernel/v/Read/ReadVariableOpReadVariableOpAdam/GO-01_mod/kernel/v*
_output_shapes

:*
dtype0

Adam/GO-02_inp/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/GO-02_inp/kernel/v

+Adam/GO-02_inp/kernel/v/Read/ReadVariableOpReadVariableOpAdam/GO-02_inp/kernel/v*
_output_shapes

:*
dtype0

Adam/GO-01_inp/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/GO-01_inp/kernel/v

+Adam/GO-01_inp/kernel/v/Read/ReadVariableOpReadVariableOpAdam/GO-01_inp/kernel/v*
_output_shapes

:*
dtype0

Adam/GO-output_mod/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameAdam/GO-output_mod/kernel/m

/Adam/GO-output_mod/kernel/m/Read/ReadVariableOpReadVariableOpAdam/GO-output_mod/kernel/m*
_output_shapes

:*
dtype0

Adam/GO-02_mod/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/GO-02_mod/bias/m
{
)Adam/GO-02_mod/bias/m/Read/ReadVariableOpReadVariableOpAdam/GO-02_mod/bias/m*
_output_shapes
:*
dtype0

Adam/GO-02_mod/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/GO-02_mod/kernel/m

+Adam/GO-02_mod/kernel/m/Read/ReadVariableOpReadVariableOpAdam/GO-02_mod/kernel/m*
_output_shapes

:*
dtype0

Adam/GO-01_mod/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/GO-01_mod/bias/m
{
)Adam/GO-01_mod/bias/m/Read/ReadVariableOpReadVariableOpAdam/GO-01_mod/bias/m*
_output_shapes
:*
dtype0

Adam/GO-01_mod/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/GO-01_mod/kernel/m

+Adam/GO-01_mod/kernel/m/Read/ReadVariableOpReadVariableOpAdam/GO-01_mod/kernel/m*
_output_shapes

:*
dtype0

Adam/GO-02_inp/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/GO-02_inp/kernel/m

+Adam/GO-02_inp/kernel/m/Read/ReadVariableOpReadVariableOpAdam/GO-02_inp/kernel/m*
_output_shapes

:*
dtype0

Adam/GO-01_inp/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/GO-01_inp/kernel/m

+Adam/GO-01_inp/kernel/m/Read/ReadVariableOpReadVariableOpAdam/GO-01_inp/kernel/m*
_output_shapes

:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

GO-output_mod/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameGO-output_mod/kernel
}
(GO-output_mod/kernel/Read/ReadVariableOpReadVariableOpGO-output_mod/kernel*
_output_shapes

:*
dtype0
t
GO-02_mod/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameGO-02_mod/bias
m
"GO-02_mod/bias/Read/ReadVariableOpReadVariableOpGO-02_mod/bias*
_output_shapes
:*
dtype0
|
GO-02_mod/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameGO-02_mod/kernel
u
$GO-02_mod/kernel/Read/ReadVariableOpReadVariableOpGO-02_mod/kernel*
_output_shapes

:*
dtype0
t
GO-01_mod/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameGO-01_mod/bias
m
"GO-01_mod/bias/Read/ReadVariableOpReadVariableOpGO-01_mod/bias*
_output_shapes
:*
dtype0
|
GO-01_mod/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameGO-01_mod/kernel
u
$GO-01_mod/kernel/Read/ReadVariableOpReadVariableOpGO-01_mod/kernel*
_output_shapes

:*
dtype0
|
GO-02_inp/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameGO-02_inp/kernel
u
$GO-02_inp/kernel/Read/ReadVariableOpReadVariableOpGO-02_inp/kernel*
_output_shapes

:*
dtype0
|
GO-01_inp/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameGO-01_inp/kernel
u
$GO-01_inp/kernel/Read/ReadVariableOpReadVariableOpGO-01_inp/kernel*
_output_shapes

:*
dtype0

NoOpNoOp
Ò=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*=
value=B= Bù<

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
term_direct_gene_map
	module_dimensions

module_children_num
gene_layers
module_layers
mod_layer_list
mod_neighbor_map
	optimizer

signatures
#_self_saveable_object_factories*
5
0
1
2
3
4
5
6*
5
0
1
2
3
4
5
6*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

 trace_0
!trace_1* 
* 
* 
* 
* 

	"GO:01
	#GO:02*
M
	$GO:01
%GO:01-output
	&GO:02
'GO:02-output
(	GO:output*

)0
*1* 
'
+	GO:output
	,GO:01
	-GO:02* 
Ð
.iter

/beta_1

0beta_2
	1decay
2learning_ratemmmmmmmvvvvvvv*

3serving_default* 
* 
PJ
VARIABLE_VALUEGO-01_inp/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEGO-02_inp/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEGO-01_mod/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEGO-01_mod/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEGO-02_mod/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEGO-02_mod/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEGO-output_mod/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
"0
#1
$2
%3
&4
'5
(6*

40*
* 
* 
* 
* 
* 
* 
Á
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
#;_self_saveable_object_factories*
Á
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
#B_self_saveable_object_factories*
Ë
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

kernel
bias
#I_self_saveable_object_factories*
y
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
#N_self_saveable_object_factories* 
Ë
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

kernel
bias
#U_self_saveable_object_factories*
y
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
#Z_self_saveable_object_factories* 
Á
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

kernel
#a_self_saveable_object_factories*
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
8
b	variables
c	keras_api
	dtotal
	ecount*

0*

0*
* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

ktrace_0* 

ltrace_0* 
* 

0*

0*
* 

mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

rtrace_0* 

strace_0* 
* 

0
1*

0
1*
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
¯
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses* 
* 

0
1*

0
1*
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
´
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses* 
* 

0*

0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 

d0
e1*

b	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
sm
VARIABLE_VALUEAdam/GO-01_inp/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/GO-02_inp/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/GO-01_mod/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/GO-01_mod/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/GO-02_mod/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/GO-02_mod/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/GO-output_mod/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/GO-01_inp/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/GO-02_inp/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/GO-01_mod/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/GO-01_mod/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/GO-02_mod/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/GO-02_mod/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/GO-output_mod/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
serving_default_args_0Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
½
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0GO-01_inp/kernelGO-02_inp/kernelGO-01_mod/kernelGO-01_mod/biasGO-02_mod/kernelGO-02_mod/biasGO-output_mod/kernel*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_951954
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$GO-01_inp/kernel/Read/ReadVariableOp$GO-02_inp/kernel/Read/ReadVariableOp$GO-01_mod/kernel/Read/ReadVariableOp"GO-01_mod/bias/Read/ReadVariableOp$GO-02_mod/kernel/Read/ReadVariableOp"GO-02_mod/bias/Read/ReadVariableOp(GO-output_mod/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/GO-01_inp/kernel/m/Read/ReadVariableOp+Adam/GO-02_inp/kernel/m/Read/ReadVariableOp+Adam/GO-01_mod/kernel/m/Read/ReadVariableOp)Adam/GO-01_mod/bias/m/Read/ReadVariableOp+Adam/GO-02_mod/kernel/m/Read/ReadVariableOp)Adam/GO-02_mod/bias/m/Read/ReadVariableOp/Adam/GO-output_mod/kernel/m/Read/ReadVariableOp+Adam/GO-01_inp/kernel/v/Read/ReadVariableOp+Adam/GO-02_inp/kernel/v/Read/ReadVariableOp+Adam/GO-01_mod/kernel/v/Read/ReadVariableOp)Adam/GO-01_mod/bias/v/Read/ReadVariableOp+Adam/GO-02_mod/kernel/v/Read/ReadVariableOp)Adam/GO-02_mod/bias/v/Read/ReadVariableOp/Adam/GO-output_mod/kernel/v/Read/ReadVariableOpConst*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_952061
á
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameGO-01_inp/kernelGO-02_inp/kernelGO-01_mod/kernelGO-01_mod/biasGO-02_mod/kernelGO-02_mod/biasGO-output_mod/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/GO-01_inp/kernel/mAdam/GO-02_inp/kernel/mAdam/GO-01_mod/kernel/mAdam/GO-01_mod/bias/mAdam/GO-02_mod/kernel/mAdam/GO-02_mod/bias/mAdam/GO-output_mod/kernel/mAdam/GO-01_inp/kernel/vAdam/GO-02_inp/kernel/vAdam/GO-01_mod/kernel/vAdam/GO-01_mod/bias/vAdam/GO-02_mod/kernel/vAdam/GO-02_mod/bias/vAdam/GO-output_mod/kernel/v*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_952155

{
'__inference_GO-01_inp_layer_call_fn_165

inputs
unknown:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_GO-01_inp_layer_call_and_return_conditional_losses_159`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á<
ÿ
__inference__traced_save_952061
file_prefix/
+savev2_go_01_inp_kernel_read_readvariableop/
+savev2_go_02_inp_kernel_read_readvariableop/
+savev2_go_01_mod_kernel_read_readvariableop-
)savev2_go_01_mod_bias_read_readvariableop/
+savev2_go_02_mod_kernel_read_readvariableop-
)savev2_go_02_mod_bias_read_readvariableop3
/savev2_go_output_mod_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_go_01_inp_kernel_m_read_readvariableop6
2savev2_adam_go_02_inp_kernel_m_read_readvariableop6
2savev2_adam_go_01_mod_kernel_m_read_readvariableop4
0savev2_adam_go_01_mod_bias_m_read_readvariableop6
2savev2_adam_go_02_mod_kernel_m_read_readvariableop4
0savev2_adam_go_02_mod_bias_m_read_readvariableop:
6savev2_adam_go_output_mod_kernel_m_read_readvariableop6
2savev2_adam_go_01_inp_kernel_v_read_readvariableop6
2savev2_adam_go_02_inp_kernel_v_read_readvariableop6
2savev2_adam_go_01_mod_kernel_v_read_readvariableop4
0savev2_adam_go_01_mod_bias_v_read_readvariableop6
2savev2_adam_go_02_mod_kernel_v_read_readvariableop4
0savev2_adam_go_02_mod_bias_v_read_readvariableop:
6savev2_adam_go_output_mod_kernel_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: «
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ô
valueÊBÇB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH§
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ï
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_go_01_inp_kernel_read_readvariableop+savev2_go_02_inp_kernel_read_readvariableop+savev2_go_01_mod_kernel_read_readvariableop)savev2_go_01_mod_bias_read_readvariableop+savev2_go_02_mod_kernel_read_readvariableop)savev2_go_02_mod_bias_read_readvariableop/savev2_go_output_mod_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_go_01_inp_kernel_m_read_readvariableop2savev2_adam_go_02_inp_kernel_m_read_readvariableop2savev2_adam_go_01_mod_kernel_m_read_readvariableop0savev2_adam_go_01_mod_bias_m_read_readvariableop2savev2_adam_go_02_mod_kernel_m_read_readvariableop0savev2_adam_go_02_mod_bias_m_read_readvariableop6savev2_adam_go_output_mod_kernel_m_read_readvariableop2savev2_adam_go_01_inp_kernel_v_read_readvariableop2savev2_adam_go_02_inp_kernel_v_read_readvariableop2savev2_adam_go_01_mod_kernel_v_read_readvariableop0savev2_adam_go_01_mod_bias_v_read_readvariableop2savev2_adam_go_02_mod_kernel_v_read_readvariableop0savev2_adam_go_02_mod_bias_v_read_readvariableop6savev2_adam_go_output_mod_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*á
_input_shapesÏ
Ì: :::::::: : : : : : : ::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::

_output_shapes
: 
¯'
ß
F__inference_restricted_nn_layer_call_and_return_conditional_losses_383

inputs3
!go_01_inp_readvariableop_resource:3
!go_02_inp_readvariableop_resource::
(go_01_mod_matmul_readvariableop_resource:7
)go_01_mod_biasadd_readvariableop_resource::
(go_02_mod_matmul_readvariableop_resource:7
)go_02_mod_biasadd_readvariableop_resource:>
,go_output_mod_matmul_readvariableop_resource:
identity¢GO-01_inp/ReadVariableOp¢ GO-01_mod/BiasAdd/ReadVariableOp¢GO-01_mod/MatMul/ReadVariableOp¢GO-02_inp/ReadVariableOp¢ GO-02_mod/BiasAdd/ReadVariableOp¢GO-02_mod/MatMul/ReadVariableOp¢#GO-output_mod/MatMul/ReadVariableOpz
GO-01_inp/ReadVariableOpReadVariableOp!go_01_inp_readvariableop_resource*
_output_shapes

:*
dtype0
GO-01_inp/mul/yConst*
_output_shapes

:*
dtype0*9
value0B."   ?          ?                y
GO-01_inp/mulMul GO-01_inp/ReadVariableOp:value:0GO-01_inp/mul/y:output:0*
T0*
_output_shapes

:g
GO-01_inp/MatMulMatMulinputsGO-01_inp/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
GO-02_inp/ReadVariableOpReadVariableOp!go_02_inp_readvariableop_resource*
_output_shapes

:*
dtype0
GO-02_inp/mul/yConst*
_output_shapes

:*
dtype0*9
value0B."                   ?          ?y
GO-02_inp/mulMul GO-02_inp/ReadVariableOp:value:0GO-02_inp/mul/y:output:0*
T0*
_output_shapes

:g
GO-02_inp/MatMulMatMulinputsGO-02_inp/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :g
concat/concatIdentityGO-01_inp/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
GO-01_mod/MatMul/ReadVariableOpReadVariableOp(go_01_mod_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
GO-01_mod/MatMulMatMulconcat/concat:output:0'GO-01_mod/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 GO-01_mod/BiasAdd/ReadVariableOpReadVariableOp)go_01_mod_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
GO-01_mod/BiasAddBiasAddGO-01_mod/MatMul:product:0(GO-01_mod/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
GO-01_mod/TanhTanhGO-01_mod/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :i
concat_1/concatIdentityGO-02_inp/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
GO-02_mod/MatMul/ReadVariableOpReadVariableOp(go_02_mod_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
GO-02_mod/MatMulMatMulconcat_1/concat:output:0'GO-02_mod/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 GO-02_mod/BiasAdd/ReadVariableOpReadVariableOp)go_02_mod_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
GO-02_mod/BiasAddBiasAddGO-02_mod/MatMul:product:0(GO-02_mod/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
GO-02_mod/TanhTanhGO-02_mod/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_2ConcatV2GO-01_mod/Tanh:y:0GO-02_mod/Tanh:y:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#GO-output_mod/MatMul/ReadVariableOpReadVariableOp,go_output_mod_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
GO-output_mod/MatMulMatMulconcat_2:output:0+GO-output_mod/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp^GO-01_inp/ReadVariableOp!^GO-01_mod/BiasAdd/ReadVariableOp ^GO-01_mod/MatMul/ReadVariableOp^GO-02_inp/ReadVariableOp!^GO-02_mod/BiasAdd/ReadVariableOp ^GO-02_mod/MatMul/ReadVariableOp$^GO-output_mod/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 m
IdentityIdentityGO-output_mod/MatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : : 24
GO-01_inp/ReadVariableOpGO-01_inp/ReadVariableOp2D
 GO-01_mod/BiasAdd/ReadVariableOp GO-01_mod/BiasAdd/ReadVariableOp2B
GO-01_mod/MatMul/ReadVariableOpGO-01_mod/MatMul/ReadVariableOp24
GO-02_inp/ReadVariableOpGO-02_inp/ReadVariableOp2D
 GO-02_mod/BiasAdd/ReadVariableOp GO-02_mod/BiasAdd/ReadVariableOp2B
GO-02_mod/MatMul/ReadVariableOpGO-02_mod/MatMul/ReadVariableOp2J
#GO-output_mod/MatMul/ReadVariableOp#GO-output_mod/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
B__inference_GO-02_mod_layer_call_and_return_conditional_losses_226

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

B__inference_GO-01_inp_layer_call_and_return_conditional_losses_159

inputs)
readvariableop_resource:
identity¢ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0v
mul/yConst*
_output_shapes

:*
dtype0*9
value0B."   ?          ?                [
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:S
MatMulMatMulinputsmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 _
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ëo
 
"__inference__traced_restore_952155
file_prefix3
!assignvariableop_go_01_inp_kernel:5
#assignvariableop_1_go_02_inp_kernel:5
#assignvariableop_2_go_01_mod_kernel:/
!assignvariableop_3_go_01_mod_bias:5
#assignvariableop_4_go_02_mod_kernel:/
!assignvariableop_5_go_02_mod_bias:9
'assignvariableop_6_go_output_mod_kernel:&
assignvariableop_7_adam_iter:	 (
assignvariableop_8_adam_beta_1: (
assignvariableop_9_adam_beta_2: (
assignvariableop_10_adam_decay: 0
&assignvariableop_11_adam_learning_rate: #
assignvariableop_12_total: #
assignvariableop_13_count: =
+assignvariableop_14_adam_go_01_inp_kernel_m:=
+assignvariableop_15_adam_go_02_inp_kernel_m:=
+assignvariableop_16_adam_go_01_mod_kernel_m:7
)assignvariableop_17_adam_go_01_mod_bias_m:=
+assignvariableop_18_adam_go_02_mod_kernel_m:7
)assignvariableop_19_adam_go_02_mod_bias_m:A
/assignvariableop_20_adam_go_output_mod_kernel_m:=
+assignvariableop_21_adam_go_01_inp_kernel_v:=
+assignvariableop_22_adam_go_02_inp_kernel_v:=
+assignvariableop_23_adam_go_01_mod_kernel_v:7
)assignvariableop_24_adam_go_01_mod_bias_v:=
+assignvariableop_25_adam_go_02_mod_kernel_v:7
)assignvariableop_26_adam_go_02_mod_bias_v:A
/assignvariableop_27_adam_go_output_mod_kernel_v:
identity_29¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9®
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ô
valueÊBÇB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHª
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B °
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_go_01_inp_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp#assignvariableop_1_go_02_inp_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_go_01_mod_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_go_01_mod_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_go_02_mod_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_go_02_mod_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp'assignvariableop_6_go_output_mod_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_iterIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp&assignvariableop_11_adam_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp+assignvariableop_14_adam_go_01_inp_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_go_02_inp_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_go_01_mod_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_go_01_mod_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_go_02_mod_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_go_02_mod_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_20AssignVariableOp/assignvariableop_20_adam_go_output_mod_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_go_01_inp_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_go_02_inp_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_go_01_mod_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_go_01_mod_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_go_02_mod_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_go_02_mod_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_27AssignVariableOp/assignvariableop_27_adam_go_output_mod_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ·
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: ¤
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


ò
A__inference_GO-01_mod_layer_call_and_return_conditional_losses_78

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¥
+__inference_restricted_nn_layer_call_fn_289

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_restricted_nn_layer_call_and_return_conditional_losses_277`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡	
¦
+__inference_restricted_nn_layer_call_fn_301
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_restricted_nn_layer_call_and_return_conditional_losses_277`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
È
¯
F__inference_GO-output_mod_layer_call_and_return_conditional_losses_209

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 _
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
£
)__inference_restored_function_body_951910

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_restricted_nn_layer_call_and_return_conditional_losses_383o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ

A__inference_GO-02_inp_layer_call_and_return_conditional_losses_67

inputs)
readvariableop_resource:
identity¢ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0v
mul/yConst*
_output_shapes

:*
dtype0*9
value0B."                   ?          ?[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:S
MatMulMatMulinputsmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 _
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

F__inference_restricted_nn_layer_call_and_return_conditional_losses_255
input_1"
go_01_inp_951388:"
go_02_inp_951391:"
go_01_mod_951396:
go_01_mod_951398:"
go_02_mod_951403:
go_02_mod_951405:&
go_output_mod_951410:
identity¢!GO-01_inp/StatefulPartitionedCall¢!GO-01_mod/StatefulPartitionedCall¢!GO-02_inp/StatefulPartitionedCall¢!GO-02_mod/StatefulPartitionedCall¢%GO-output_mod/StatefulPartitionedCallÞ
!GO-01_inp/StatefulPartitionedCallStatefulPartitionedCallinput_1go_01_inp_951388*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_GO-01_inp_layer_call_and_return_conditional_losses_159Þ
!GO-02_inp/StatefulPartitionedCallStatefulPartitionedCallinput_1go_02_inp_951391*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_GO-02_inp_layer_call_and_return_conditional_losses_196S
concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :w
concat/concatIdentity*GO-01_inp/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!GO-01_mod/StatefulPartitionedCallStatefulPartitionedCallconcat/concat:output:0go_01_mod_951396go_01_mod_951398*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GO-01_mod_layer_call_and_return_conditional_losses_78U
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :y
concat_1/concatIdentity*GO-02_inp/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!GO-02_mod/StatefulPartitionedCallStatefulPartitionedCallconcat_1/concat:output:0go_02_mod_951403go_02_mod_951405*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_GO-02_mod_layer_call_and_return_conditional_losses_226O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :¿
concat_2ConcatV2*GO-01_mod/StatefulPartitionedCall:output:0*GO-02_mod/StatefulPartitionedCall:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
%GO-output_mod/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0go_output_mod_951410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_GO-output_mod_layer_call_and_return_conditional_losses_209þ
NoOpNoOp"^GO-01_inp/StatefulPartitionedCall"^GO-01_mod/StatefulPartitionedCall"^GO-02_inp/StatefulPartitionedCall"^GO-02_mod/StatefulPartitionedCall&^GO-output_mod/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 }
IdentityIdentity.GO-output_mod/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : : 2F
!GO-01_inp/StatefulPartitionedCall!GO-01_inp/StatefulPartitionedCall2F
!GO-01_mod/StatefulPartitionedCall!GO-01_mod/StatefulPartitionedCall2F
!GO-02_inp/StatefulPartitionedCall!GO-02_inp/StatefulPartitionedCall2F
!GO-02_mod/StatefulPartitionedCall!GO-02_mod/StatefulPartitionedCall2N
%GO-output_mod/StatefulPartitionedCall%GO-output_mod/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ò

B__inference_GO-01_inp_layer_call_and_return_conditional_losses_101

inputs)
readvariableop_resource:
identity¢ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0v
mul/yConst*
_output_shapes

:*
dtype0*9
value0B."   ?          ?                [
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:S
MatMulMatMulinputsmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 _
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

{
'__inference_GO-02_inp_layer_call_fn_202

inputs
unknown:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_GO-02_inp_layer_call_and_return_conditional_losses_196`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

F__inference_restricted_nn_layer_call_and_return_conditional_losses_277

inputs"
go_01_inp_951237:"
go_02_inp_951250:"
go_01_mod_951267:
go_01_mod_951269:"
go_02_mod_951286:
go_02_mod_951288:&
go_output_mod_951301:
identity¢!GO-01_inp/StatefulPartitionedCall¢!GO-01_mod/StatefulPartitionedCall¢!GO-02_inp/StatefulPartitionedCall¢!GO-02_mod/StatefulPartitionedCall¢%GO-output_mod/StatefulPartitionedCallÝ
!GO-01_inp/StatefulPartitionedCallStatefulPartitionedCallinputsgo_01_inp_951237*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_GO-01_inp_layer_call_and_return_conditional_losses_159Ý
!GO-02_inp/StatefulPartitionedCallStatefulPartitionedCallinputsgo_02_inp_951250*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_GO-02_inp_layer_call_and_return_conditional_losses_196S
concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :w
concat/concatIdentity*GO-01_inp/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!GO-01_mod/StatefulPartitionedCallStatefulPartitionedCallconcat/concat:output:0go_01_mod_951267go_01_mod_951269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_GO-01_mod_layer_call_and_return_conditional_losses_78U
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :y
concat_1/concatIdentity*GO-02_inp/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!GO-02_mod/StatefulPartitionedCallStatefulPartitionedCallconcat_1/concat:output:0go_02_mod_951286go_02_mod_951288*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_GO-02_mod_layer_call_and_return_conditional_losses_226O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :¿
concat_2ConcatV2*GO-01_mod/StatefulPartitionedCall:output:0*GO-02_mod/StatefulPartitionedCall:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
%GO-output_mod/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0go_output_mod_951301*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_GO-output_mod_layer_call_and_return_conditional_losses_209þ
NoOpNoOp"^GO-01_inp/StatefulPartitionedCall"^GO-01_mod/StatefulPartitionedCall"^GO-02_inp/StatefulPartitionedCall"^GO-02_mod/StatefulPartitionedCall&^GO-output_mod/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 }
IdentityIdentity.GO-output_mod/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : : 2F
!GO-01_inp/StatefulPartitionedCall!GO-01_inp/StatefulPartitionedCall2F
!GO-01_mod/StatefulPartitionedCall!GO-01_mod/StatefulPartitionedCall2F
!GO-02_inp/StatefulPartitionedCall!GO-02_inp/StatefulPartitionedCall2F
!GO-02_mod/StatefulPartitionedCall!GO-02_mod/StatefulPartitionedCall2N
%GO-output_mod/StatefulPartitionedCall%GO-output_mod/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò

$__inference_signature_wrapper_951954

args_0
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_951927o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
é

ø
!__inference__wrapped_model_951927

args_0&
restricted_nn_951911:&
restricted_nn_951913:&
restricted_nn_951915:"
restricted_nn_951917:&
restricted_nn_951919:"
restricted_nn_951921:&
restricted_nn_951923:
identity¢%restricted_nn/StatefulPartitionedCallÜ
%restricted_nn/StatefulPartitionedCallStatefulPartitionedCallargs_0restricted_nn_951911restricted_nn_951913restricted_nn_951915restricted_nn_951917restricted_nn_951919restricted_nn_951921restricted_nn_951923*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *2
f-R+
)__inference_restored_function_body_951910}
IdentityIdentity.restricted_nn/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
NoOpNoOp&^restricted_nn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : : 2N
%restricted_nn/StatefulPartitionedCall%restricted_nn/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
Ò

B__inference_GO-02_inp_layer_call_and_return_conditional_losses_196

inputs)
readvariableop_resource:
identity¢ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0v
mul/yConst*
_output_shapes

:*
dtype0*9
value0B."                   ?          ?[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:S
MatMulMatMulinputsmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 _
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*©
serving_default
9
args_0/
serving_default_args_0:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:w

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
term_direct_gene_map
	module_dimensions

module_children_num
gene_layers
module_layers
mod_layer_list
mod_neighbor_map
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_model
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
º
trace_0
trace_12
+__inference_restricted_nn_layer_call_fn_301
+__inference_restricted_nn_layer_call_fn_289¦
²
FullArgSpec
args
jinputs
jprune
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
ð
 trace_0
!trace_12¹
F__inference_restricted_nn_layer_call_and_return_conditional_losses_383
F__inference_restricted_nn_layer_call_and_return_conditional_losses_255¦
²
FullArgSpec
args
jinputs
jprune
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0z!trace_1
ËBÈ
!__inference__wrapped_model_951927args_0"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
6
	"GO:01
	#GO:02"
trackable_dict_wrapper
i
	$GO:01
%GO:01-output
	&GO:02
'GO:02-output
(	GO:output"
trackable_dict_wrapper
.
)0
*1"
trackable_list_wrapper
E
+	GO:output
	,GO:01
	-GO:02"
trackable_dict_wrapper
ß
.iter

/beta_1

0beta_2
	1decay
2learning_ratemmmmmmmvvvvvvv"
	optimizer
,
3serving_default"
signature_map
 "
trackable_dict_wrapper
": 2GO-01_inp/kernel
": 2GO-02_inp/kernel
": 2GO-01_mod/kernel
:2GO-01_mod/bias
": 2GO-02_mod/kernel
:2GO-02_mod/bias
&:$2GO-output_mod/kernel
 "
trackable_list_wrapper
Q
"0
#1
$2
%3
&4
'5
(6"
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
äBá
+__inference_restricted_nn_layer_call_fn_301input_1"¦
²
FullArgSpec
args
jinputs
jprune
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ãBà
+__inference_restricted_nn_layer_call_fn_289inputs"¦
²
FullArgSpec
args
jinputs
jprune
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
F__inference_restricted_nn_layer_call_and_return_conditional_losses_383inputs"¦
²
FullArgSpec
args
jinputs
jprune
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
F__inference_restricted_nn_layer_call_and_return_conditional_losses_255input_1"¦
²
FullArgSpec
args
jinputs
jprune
varargs
 
varkw
 
defaults¢
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
#;_self_saveable_object_factories"
_tf_keras_layer
Ö
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
#B_self_saveable_object_factories"
_tf_keras_layer
à
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

kernel
bias
#I_self_saveable_object_factories"
_tf_keras_layer

J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
#N_self_saveable_object_factories"
_tf_keras_layer
à
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

kernel
bias
#U_self_saveable_object_factories"
_tf_keras_layer

V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
#Z_self_saveable_object_factories"
_tf_keras_layer
Ö
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

kernel
#a_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÊBÇ
$__inference_signature_wrapper_951954args_0"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
b	variables
c	keras_api
	dtotal
	ecount"
_tf_keras_metric
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
á
ktrace_02Ä
'__inference_GO-01_inp_layer_call_fn_165
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zktrace_0
ü
ltrace_02ß
B__inference_GO-01_inp_layer_call_and_return_conditional_losses_101
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zltrace_0
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
á
rtrace_02Ä
'__inference_GO-02_inp_layer_call_fn_202
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zrtrace_0
û
strace_02Þ
A__inference_GO-02_inp_layer_call_and_return_conditional_losses_67
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zstrace_0
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ë
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ð
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
.
d0
e1"
trackable_list_wrapper
-
b	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÑBÎ
'__inference_GO-01_inp_layer_call_fn_165inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ìBé
B__inference_GO-01_inp_layer_call_and_return_conditional_losses_101inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÑBÎ
'__inference_GO-02_inp_layer_call_fn_202inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ëBè
A__inference_GO-02_inp_layer_call_and_return_conditional_losses_67inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
':%2Adam/GO-01_inp/kernel/m
':%2Adam/GO-02_inp/kernel/m
':%2Adam/GO-01_mod/kernel/m
!:2Adam/GO-01_mod/bias/m
':%2Adam/GO-02_mod/kernel/m
!:2Adam/GO-02_mod/bias/m
+:)2Adam/GO-output_mod/kernel/m
':%2Adam/GO-01_inp/kernel/v
':%2Adam/GO-02_inp/kernel/v
':%2Adam/GO-01_mod/kernel/v
!:2Adam/GO-01_mod/bias/v
':%2Adam/GO-02_mod/kernel/v
!:2Adam/GO-02_mod/bias/v
+:)2Adam/GO-output_mod/kernel/v¡
B__inference_GO-01_inp_layer_call_and_return_conditional_losses_101[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
'__inference_GO-01_inp_layer_call_fn_165N/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ 
A__inference_GO-02_inp_layer_call_and_return_conditional_losses_67[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
'__inference_GO-02_inp_layer_call_fn_202N/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
!__inference__wrapped_model_951927o/¢,
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ°
F__inference_restricted_nn_layer_call_and_return_conditional_losses_255f4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¯
F__inference_restricted_nn_layer_call_and_return_conditional_losses_383e3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_restricted_nn_layer_call_fn_289X3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_restricted_nn_layer_call_fn_301Y4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¡
$__inference_signature_wrapper_951954y9¢6
¢ 
/ª,
*
args_0 
args_0ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ