from .deep_learning_controller import deep_learning_controller

# Blueprint를 모아서 메인 애플리케이션에 등록
def register_blueprints(app):
    app.register_blueprint(deep_learning_controller, url_prefix='/api')