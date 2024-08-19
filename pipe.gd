extends Area2D


# Declare member variables here. Examples:
# var a = 2
var velocity = Vector2(-190, 0)
var score = false

signal scored


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	position += velocity * delta
	if not score and position.x <= 250:
		score = true
		print("scored!")
		emit_signal("scored")

func change_color():
	for child in get_children():
		if child is Sprite:
			child.modulate = Color(40 / 255.0, 219 / 255.0, 149 / 255.0)

func _on_VisibilityNotifier2D_screen_exited():
	queue_free()


