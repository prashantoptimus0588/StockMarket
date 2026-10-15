from database.connection import Base, engine
from database.models import Assessment  # Registers the model


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Assessment database tables created successfully.")


if __name__ == "__main__":
    init_db()
