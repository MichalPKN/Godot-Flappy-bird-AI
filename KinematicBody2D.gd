extends KinematicBody2D

signal hit

var velocity = Vector2(0, 20)
var screen_size

# Declare member variables here. Examples:
# var a = 2
# var b = "text"
signal game_over

# Called when the node enters the scene tree for the first time.
func _ready():
	randomize()
	screen_size = get_viewport_rect().size


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	move_and_slide(velocity)
	#velocity.y += 650 * delta
	velocity.y += 680 * delta
	if Input.is_action_just_pressed("ui_up"):
		jump()
	position.y += velocity.y * delta
	if position.y > screen_size.y or position.y < 0:
		emit_signal("game_over")
		
		

func jump():
	velocity.y = -240

#func _on_pipe_body_entered(body):
	#print(body.name)
#	hide()
	#emit_signal("game_over")

