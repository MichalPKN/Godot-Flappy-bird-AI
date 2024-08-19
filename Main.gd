extends Node
signal hit

#export (PackedScene) var pipe
# Declare member variables here. Examples:
# var a = 2
# var b = "text"

var python_script = NNpython
onready var pipe = preload("res://pipe.tscn")
onready var player = get_node("player")
onready var pipe_timer = $PipeTimer
var done = false
var reward = 0
var score = 0
var old_state = false
var action = null
var del = 0.033333

# Called when the node enters the scene tree for the first time.
func _ready():
	#Engine.time_scale = 3
	#Engine.target_fps = 90
	randomize()
	spawn_pipe()
	pipe_timer.start(1.6)
	player.connect("game_over", self, "_on_game_over")


# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass


func _on_PipeTimer_timeout():
	spawn_pipe()


func _on_player_game_over():
	$PipeTimer.stop()
	
func _process(delta):
	reward += 0.05
	if python_script.training: # and Engine.get_frames_drawn() % 4 == 0:
		var state = get_current_state()
		#print(reward, state, done)
		if old_state:
			python_script.train_short_memory(old_state, action, reward, state, done)
			python_script.remember(old_state, action, reward, state, done)
		if done == true:
			print(state)
			python_script.done(score)
			get_tree().reload_current_scene()
		action = python_script.get_action(state)
		if action:
			player.jump()

		old_state = state
	

func get_current_state():
	var player_y = player.position.y
	var player_vel_y = player.velocity.y
	var c_pipe = get_closest_pipe()
	var c_pipe_distance = c_pipe.position.x - player.position.x
	var c_pipe_y = c_pipe.position.y
	#return [player_y, player_vel_y, c_pipe_distance, c_pipe_y]
	var c_pipe_top_dist = player_y - c_pipe_y - 88
	var c_pipe_bottom_dist = c_pipe_y + 88 - player_y
	return [player_vel_y, c_pipe_distance + 50, c_pipe_top_dist-21, c_pipe_bottom_dist-21]

func get_closest_pipe():
	var closest_distance = INF
	var closest_pipe = null
	for p in get_tree().get_nodes_in_group('pipe_group'):
		if p.position.x + 50 > player.position.x: #prev: + 50
			var distance = p.position.x - player.position.x
			if distance < closest_distance:
				closest_distance = distance
				closest_pipe = p
	closest_pipe.change_color()
	return closest_pipe

func _on_pipe_body_entered(body):
	died()

func _on_game_over():
	died()

func died():
	if not done:
		done = true
		reward -= 100
		print("RIP", reward)

func _on_scored():
	reward += 20
	score += 1
	print(score)



func _on_Timer_timeout():
	pass
	#reward += 1

func spawn_pipe():
	var Pipe = pipe.instance()
	add_child(Pipe)
	Pipe.position = Vector2(800, rand_range(160, 430))
	Pipe.connect("body_entered", self, "_on_pipe_body_entered")
	Pipe.connect("scored", self, "_on_scored")



